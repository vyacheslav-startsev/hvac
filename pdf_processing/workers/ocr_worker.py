"""
Воркер для обработки графических страниц с помощью OCR (только PaddleOCR)
"""
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from redis import Redis
import logging
import os
from typing import Optional
from config import Config
from .. import TaskTracker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация OCR движка (глобально для переиспользования)
_ocr_engine = None


def get_ocr_engine():
    """Получить инициализированный PaddleOCR движок"""
    global _ocr_engine

    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang='ru',
            use_gpu=Config.OCR_USE_GPU,
            show_log=False
        )
        logger.info("Инициализирован PaddleOCR")
    except ImportError:
        logger.error("PaddleOCR не установлен. Установите: pip install paddleocr")
        raise

    return _ocr_engine


def extract_text_with_paddleocr(image_path: str) -> str:
    """Извлечь текст с помощью PaddleOCR"""
    ocr = get_ocr_engine()

    try:
        result = ocr.ocr(image_path, cls=True)

        # Собрать весь текст
        text_lines = []
        if result and result[0]:
            for line in result[0]:
                if len(line) >= 2:
                    text_lines.append(line[1][0])  # line[1][0] содержит текст

        return '\n'.join(text_lines)

    except Exception as e:
        logger.error(f"Ошибка PaddleOCR: {e}")
        raise


def convert_page_to_image(pdf_path: str, page_num: int) -> str:
    """
    Конвертировать страницу PDF в изображение

    Args:
        pdf_path: Путь к PDF
        page_num: Номер страницы (с 0)

    Returns:
        Путь к временному изображению
    """
    try:
        Config.ensure_dirs()

        # Конвертировать только нужную страницу
        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=300  # Высокое разрешение для лучшего OCR
        )

        if not images:
            raise ValueError(f"Не удалось конвертировать страницу {page_num}")

        # Сохранить временное изображение
        image = images[0]
        temp_image_path = os.path.join(
            Config.TEMP_DIR,
            f"page_{page_num}_{os.getpid()}.png"
        )
        image.save(temp_image_path, 'PNG')

        return temp_image_path

    except Exception as e:
        logger.error(f"Ошибка конвертации страницы в изображение: {e}")
        raise


def process_ocr_page(pdf_path: str, page_num: int, job_id: str):
    """
    Обработать графическую страницу с помощью OCR (функция для RQ воркера)

    Args:
        pdf_path: Путь к PDF файлу
        page_num: Номер страницы
        job_id: ID задачи
    """
    logger.info(f"[{job_id}] Обработка OCR страницы {page_num}")

    temp_image_path = None

    try:
        # Подключиться к Redis
        redis_client = Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB,
            password=Config.REDIS_PASSWORD
        )
        tracker = TaskTracker(redis_client)

        # Конвертировать страницу в изображение
        logger.info(f"[{job_id}] Конвертация страницы {page_num} в изображение...")
        temp_image_path = convert_page_to_image(pdf_path, page_num)

        # Выполнить OCR
        logger.info(f"[{job_id}] Запуск PaddleOCR для страницы {page_num}...")
        text = extract_text_with_paddleocr(temp_image_path)

        # Отметить как завершенную
        all_completed = tracker.mark_page_completed(
            job_id, page_num, text, success=True
        )

        if all_completed:
            logger.info(f"[{job_id}] Все страницы обработаны!")

        logger.info(
            f"[{job_id}] Страница {page_num} обработана (OCR): "
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
            f"[{job_id}] Ошибка OCR обработки страницы {page_num}: {e}",
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

    finally:
        # Удалить временное изображение
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except:
                pass


if __name__ == '__main__':
    """Запуск воркера для OCR страниц"""
    from rq import Worker, Queue

    redis_conn = Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        db=Config.REDIS_DB,
        password=Config.REDIS_PASSWORD
    )

    queue = Queue(Config.OCR_QUEUE, connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    logger.info(f"Запуск воркера OCR страниц (очередь: {Config.OCR_QUEUE})")
    worker.work()
