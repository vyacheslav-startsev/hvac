"""
Сборка финальных результатов распознавания PDF
Результаты возвращаются как строка и логируются, без сохранения в файл
"""
from typing import Dict, Optional
import logging
from config import Config
from .tracker import TaskTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResultAssembler:
    """Сборка результатов распознавания в единый документ"""

    def __init__(self, tracker: TaskTracker):
        self.tracker = tracker

    def assemble_results(self, job_id: str,
                        output_format: str = 'txt') -> Optional[str]:
        """
        Собрать все результаты в единый текст

        Args:
            job_id: ID задачи
            output_format: Формат вывода ('txt', 'md', 'json')

        Returns:
            Собранный текст или None при ошибке
        """
        logger.info(f"[{job_id}] Начало сборки результатов")

        try:
            # Получить статус задачи
            job_status = self.tracker.get_job_status(job_id)

            if not job_status:
                logger.error(f"Задача {job_id} не найдена")
                return None

            # Проверить статус
            if job_status['status'] not in ['completed', 'completed_with_errors']:
                logger.warning(
                    f"Задача {job_id} еще не завершена "
                    f"(статус: {job_status['status']})"
                )
                return None

            # Получить все результаты
            results = self.tracker.get_results(job_id)

            if not results:
                logger.warning(f"Нет результатов для задачи {job_id}")
                return None

            total_pages = int(job_status['total_pages'])

            # Собрать текст по порядку страниц
            assembled_text = []
            missing_pages = []

            for page_num in range(total_pages):
                if page_num in results:
                    page_text = results[page_num]
                    assembled_text.append(
                        self._format_page(page_num, page_text, output_format)
                    )
                else:
                    missing_pages.append(page_num)
                    assembled_text.append(
                        self._format_missing_page(page_num, output_format)
                    )

            # Создать финальный текст
            if output_format == 'json':
                import json
                result_data = {
                    'job_id': job_id,
                    'pdf_path': job_status['pdf_path'],
                    'total_pages': total_pages,
                    'completed_pages': int(job_status['completed_pages']),
                    'failed_pages': int(job_status['failed_pages']),
                    'missing_pages': missing_pages,
                    'pages': results
                }
                final_text = json.dumps(result_data, ensure_ascii=False, indent=2)
            else:
                # Добавить заголовок
                header = self._create_header(job_status, output_format)
                final_text = header + '\n\n' + '\n\n'.join(assembled_text)

            # Логировать результат
            logger.info(
                f"[{job_id}] Результаты собраны: "
                f"{total_pages} страниц, {len(missing_pages)} пропущено"
            )
            logger.info(f"[{job_id}] Общая длина текста: {len(final_text)} символов")

            # Логировать первые 1000 символов для просмотра
            logger.info(f"[{job_id}] Начало результата:\n{final_text[:1000]}...")

            return final_text

        except Exception as e:
            logger.error(f"[{job_id}] Ошибка сборки результатов: {e}", exc_info=True)
            return None

    def _format_page(self, page_num: int, text: str,
                    output_format: str) -> str:
        """Форматировать текст страницы"""
        if output_format == 'md':
            return f"## Страница {page_num + 1}\n\n{text}"
        elif output_format == 'txt':
            return f"{'=' * 60}\nСТРАНИЦА {page_num + 1}\n{'=' * 60}\n\n{text}"
        else:
            return text

    def _format_missing_page(self, page_num: int, output_format: str) -> str:
        """Форматировать пропущенную страницу"""
        if output_format == 'md':
            return f"## Страница {page_num + 1}\n\n*[Ошибка обработки]*"
        elif output_format == 'txt':
            return f"{'=' * 60}\nСТРАНИЦА {page_num + 1}\n{'=' * 60}\n\n[ОШИБКА ОБРАБОТКИ]"
        else:
            return "[ОШИБКА]"

    def _create_header(self, job_status: Dict, output_format: str) -> str:
        """Создать заголовок документа"""
        if output_format == 'md':
            return f"""# Результат распознавания PDF

**Файл:** {job_status['pdf_path']}
**ID задачи:** {job_status['job_id']}
**Всего страниц:** {job_status['total_pages']}
**Обработано успешно:** {job_status['completed_pages']}
**Ошибок:** {job_status['failed_pages']}
**Статус:** {job_status['status']}

---
"""
        else:
            return f"""{'=' * 80}
РЕЗУЛЬТАТ РАСПОЗНАВАНИЯ PDF
{'=' * 80}

Файл: {job_status['pdf_path']}
ID задачи: {job_status['job_id']}
Всего страниц: {job_status['total_pages']}
Обработано успешно: {job_status['completed_pages']}
Ошибок: {job_status['failed_pages']}
Статус: {job_status['status']}

{'=' * 80}
"""
