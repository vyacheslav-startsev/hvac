"""
Система отслеживания выполнения распределенных задач в Redis
"""
import redis
import json
import time
import uuid
from typing import Dict, Optional, List
from config import Config


class TaskTracker:
    """Отслеживание прогресса обработки PDF документов"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def create_job(self, pdf_path: str, total_pages: int) -> str:
        """
        Создать новую задачу обработки PDF

        Args:
            pdf_path: Путь к PDF файлу
            total_pages: Общее количество страниц

        Returns:
            job_id: Уникальный идентификатор задачи
        """
        job_id = str(uuid.uuid4())
        job_key = f"job:{job_id}"

        job_data = {
            'job_id': job_id,
            'pdf_path': pdf_path,
            'total_pages': total_pages,
            'completed_pages': 0,
            'failed_pages': 0,
            'status': 'pending',  # pending, processing, completed, failed
            'created_at': time.time(),
            'text_pages': 0,
            'ocr_pages': 0,
            'errors': []
        }

        # Сохранить метаданные задачи
        self.redis.hset(job_key, mapping={
            k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            for k, v in job_data.items()
        })

        # Установить TTL - 24 часа
        self.redis.expire(job_key, 86400)

        # Создать пустой список для результатов страниц
        results_key = f"job:{job_id}:results"
        self.redis.delete(results_key)  # Очистить, если существует
        self.redis.expire(results_key, 86400)

        return job_id

    def register_page_task(self, job_id: str, page_num: int,
                          page_type: str, task_id: str):
        """
        Зарегистрировать задачу обработки страницы

        Args:
            job_id: ID задачи
            page_num: Номер страницы
            page_type: Тип страницы ('text' или 'ocr')
            task_id: ID задачи в очереди RQ
        """
        page_key = f"job:{job_id}:page:{page_num}"
        page_data = {
            'page_num': page_num,
            'page_type': page_type,
            'task_id': task_id,
            'status': 'queued',
            'created_at': time.time()
        }

        self.redis.hset(page_key, mapping={
            k: str(v) for k, v in page_data.items()
        })
        self.redis.expire(page_key, 86400)

        # Увеличить счетчик страниц по типу
        job_key = f"job:{job_id}"
        if page_type == 'text':
            self.redis.hincrby(job_key, 'text_pages', 1)
        else:
            self.redis.hincrby(job_key, 'ocr_pages', 1)

    def mark_page_completed(self, job_id: str, page_num: int,
                           text: str, success: bool = True,
                           error: Optional[str] = None) -> bool:
        """
        Отметить страницу как обработанную

        Args:
            job_id: ID задачи
            page_num: Номер страницы
            text: Распознанный текст
            success: Успешна ли обработка
            error: Сообщение об ошибке (если есть)

        Returns:
            True если все страницы обработаны
        """
        job_key = f"job:{job_id}"
        page_key = f"job:{job_id}:page:{page_num}"
        results_key = f"job:{job_id}:results"

        # Обновить статус страницы
        self.redis.hset(page_key, 'status', 'completed' if success else 'failed')
        self.redis.hset(page_key, 'completed_at', time.time())

        # Сохранить результат
        if success:
            self.redis.hset(results_key, str(page_num), text)
            self.redis.hincrby(job_key, 'completed_pages', 1)
        else:
            self.redis.hincrby(job_key, 'failed_pages', 1)
            if error:
                # Добавить ошибку в список
                errors = json.loads(self.redis.hget(job_key, 'errors') or '[]')
                errors.append({
                    'page': page_num,
                    'error': error,
                    'timestamp': time.time()
                })
                self.redis.hset(job_key, 'errors', json.dumps(errors))

        # Проверить, все ли страницы обработаны
        total_pages = int(self.redis.hget(job_key, 'total_pages'))
        completed = int(self.redis.hget(job_key, 'completed_pages'))
        failed = int(self.redis.hget(job_key, 'failed_pages'))

        if completed + failed >= total_pages:
            status = 'completed' if failed == 0 else 'completed_with_errors'
            self.redis.hset(job_key, 'status', status)
            self.redis.hset(job_key, 'completed_at', time.time())
            return True

        return False

    def get_job_status(self, job_id: str) -> Dict:
        """Получить статус задачи"""
        job_key = f"job:{job_id}"
        job_data = self.redis.hgetall(job_key)

        if not job_data:
            return None

        # Декодировать байты и распарсить JSON
        result = {}
        for k, v in job_data.items():
            k = k.decode('utf-8') if isinstance(k, bytes) else k
            v = v.decode('utf-8') if isinstance(v, bytes) else v

            if k in ['errors']:
                result[k] = json.loads(v)
            else:
                result[k] = v

        return result

    def get_results(self, job_id: str) -> Dict[int, str]:
        """
        Получить все результаты обработки

        Returns:
            Словарь {номер_страницы: текст}
        """
        results_key = f"job:{job_id}:results"
        raw_results = self.redis.hgetall(results_key)

        # Преобразовать в словарь {int: str}
        results = {}
        for page_num, text in raw_results.items():
            page_num = int(page_num.decode('utf-8') if isinstance(page_num, bytes) else page_num)
            text = text.decode('utf-8') if isinstance(text, bytes) else text
            results[page_num] = text

        return results

    def wait_for_completion(self, job_id: str,
                           timeout: int = 3600,
                           poll_interval: int = 2) -> bool:
        """
        Ждать завершения задачи

        Args:
            job_id: ID задачи
            timeout: Таймаут в секундах
            poll_interval: Интервал опроса в секундах

        Returns:
            True если задача завершена успешно
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)

            if not status:
                return False

            if status['status'] in ['completed', 'completed_with_errors']:
                return True

            if status['status'] == 'failed':
                return False

            time.sleep(poll_interval)

        return False  # Таймаут

    def cleanup_job(self, job_id: str):
        """Удалить все данные задачи из Redis"""
        job_key = f"job:{job_id}"
        results_key = f"job:{job_id}:results"

        # Получить все страницы
        job_data = self.get_job_status(job_id)
        if job_data:
            total_pages = int(job_data.get('total_pages', 0))

            # Удалить все ключи страниц
            page_keys = [f"job:{job_id}:page:{i}" for i in range(total_pages)]
            if page_keys:
                self.redis.delete(*page_keys)

        # Удалить основные ключи
        self.redis.delete(job_key, results_key)
