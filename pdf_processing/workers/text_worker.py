"""
Воркер для обработки текстовых страниц PDF
"""
import fitz  # PyMuPDF
from redis import Redis
import logging
from config import Config
from .. import TaskTracker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_text_from_page(pdf_path: str, page_num: int) -> str:
    """
    Извлечь текст из текстовой страницы PDF

    Args:
        pdf_path: Путь к PDF файлу
        page_num: Номер страницы (с 0)

    Returns:
        Извлеченный текст
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]

        # Извлечь текст
        text = page.get_text()

        # Очистить и нормализовать
        text = text.strip()

        doc.close()

        return text

    except Exception as e:
        logger.error(f"Ошибка извлечения текста со страницы {page_num}: {e}")
        raise


def process_text_page(pdf_path: str, page_num: int, job_id: str):
    """
    Обработать текстовую страницу (функция для RQ воркера)

    Args:
        pdf_path: Путь к PDF файлу
        page_num: Номер страницы
        job_id: ID задачи
    """
    logger.info(f"[{job_id}] Обработка текстовой страницы {page_num}")

    try:
        # Подключиться к Redis
        redis_client = Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD
        )
        tracker = TaskTracker(redis_client)

        # Извлечь текст
        text = extract_text_from_page(pdf_path, page_num)

        # Отметить как завершенную
        all_completed = tracker.mark_page_completed(
            job_id, page_num, text, success=True
        )

        if all_completed:
            logger.info(f"[{job_id}] Все страницы обработаны!")

        logger.info(
            f"[{job_id}] Страница {page_num} обработана: "
            f"{len(text)} символов"
        )

        return {
            'page_num': page_num,
            'text_length': len(text),
            'status': 'success',
            'text': text
        }

    except Exception as e:
        logger.error(
            f"[{job_id}] Ошибка обработки страницы {page_num}: {e}",
            exc_info=True
        )

        # Отметить как неудачную
        try:
            redis_client = Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD
            )
            tracker = TaskTracker(redis_client)
            tracker.mark_page_completed(
                job_id, page_num, "", success=False, error=str(e)
            )
        except:
            pass

        raise


if __name__ == '__main__':
    """Запуск воркера для текстовых страниц"""
    from rq import Worker, Queue

    redis_conn = Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        db=Config.REDIS_DB,
        password=Config.REDIS_PASSWORD
    )

    queue = Queue(Config.TEXT_QUEUE, connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    logger.info(f"Запуск воркера текстовых страниц (очередь: {Config.TEXT_QUEUE})")
    worker.work()
