"""
Координатор обработки PDF - анализ, классификация и постановка задач в очередь
"""
import fitz  # PyMuPDF
from redis import Redis
from rq import Queue, Retry
from typing import Tuple, Dict
import logging
from config import Config
from .tracker import TaskTracker

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFCoordinator:
    """Координация обработки PDF документов"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.text_queue = Queue(Config.TEXT_QUEUE, connection=redis_client)
        self.ocr_queue = Queue(Config.OCR_QUEUE, connection=redis_client)
        self.tracker = TaskTracker(redis_client)
        Config.ensure_dirs()

    def classify_page(self, page: fitz.Page) -> str:
        """
        Классифицировать страницу PDF как текстовую или графическую

        Args:
            page: Страница PyMuPDF

        Returns:
            'text', 'image', 'mixed', или 'blank'
        """
        try:
            image_area = 0.0
            text_area = 0.0

            # Анализ блоков на странице
            blocks = page.get_text("blocks")
            
            if len(blocks) == 0 and (len(page.get_images()) > 0 or len(page.get_drawings()) > 0):
                return 'image'

            for block in blocks:
                # block[:4] - координаты (x0, y0, x1, y1)
                # block[6] - тип: 0 = текст, 1 = изображение
                rect = fitz.Rect(block[:4])
                block_area = abs(rect)

                if len(block) > 6:
                    if block[6] == 1:  # Изображение
                        image_area += block_area
                    elif block[6] == 0:  # Текст
                        text_area += block_area

            # Площадь страницы
            page_area = abs(page.rect)

            if page_area == 0:
                return 'blank'

            # Расчет покрытия
            text_coverage = text_area / page_area
            image_coverage = image_area / page_area

            # Классификация по порогам
            if image_coverage > Config.IMAGE_COVERAGE_THRESHOLD and \
               text_coverage < Config.TEXT_COVERAGE_THRESHOLD:
                return 'image'
            elif text_coverage > Config.TEXT_COVERAGE_THRESHOLD and \
                 image_coverage < Config.IMAGE_COVERAGE_THRESHOLD :
                return 'text'
            elif text_coverage == 0 and image_coverage == 0:
                return 'blank'
            else:
                return 'mixed'

        except Exception as e:
            logger.error(f"Ошибка классификации страницы: {e}")
            return 'unknown'

    def has_extractable_text(self, page: fitz.Page, min_chars: int = 50) -> bool:
        """
        Проверить, есть ли на странице извлекаемый текст

        Args:
            page: Страница PyMuPDF
            min_chars: Минимальное количество символов для текстовой страницы

        Returns:
            True если есть достаточно текста
        """
        try:
            text = page.get_text().strip()
            return len(text) >= min_chars
        except:
            return False

    def is_scanned_page(self, page: fitz.Page) -> bool:
        """
        Определить, является ли страница отсканированным изображением

        Args:
            page: Страница PyMuPDF

        Returns:
            True если страница - скан
        """
        try:
            images = page.get_images(full=True)

            if not images:
                return False

            # Проверить, занимает ли изображение всю страницу
            page_area = abs(page.rect)

            for img_info in images:
                try:
                    # img_info[0] - xref изображения
                    xref = img_info[0]

                    # Получить все вхождения изображения на странице
                    img_rects = page.get_image_rects(xref)

                    if img_rects:
                        # Проверить каждый прямоугольник
                        for img_rect in img_rects:
                            img_area = abs(img_rect)
                            coverage = img_area / page_area if page_area > 0 else 0

                            if coverage >= 0.95:  # 95% покрытия
                                return True
                except:
                    continue

            return False

        except Exception as e:
            logger.error(f"Ошибка проверки сканированной страницы: {e}")
            return False

    def determine_page_type(self, page: fitz.Page) -> Tuple[str, str]:
        """
        Определить тип страницы и очередь для обработки

        Args:
            page: Страница PyMuPDF

        Returns:
            (page_type, queue_name): тип страницы и имя очереди
        """
        # Сначала проверим, есть ли текст
        has_text = self.has_extractable_text(page)

        # Проверим, это скан
        is_scanned = self.is_scanned_page(page)

        # Классифицируем по структуре
        classification = self.classify_page(page)

        # Логика определения типа
        if is_scanned or classification == 'image':
            return ('ocr', Config.OCR_QUEUE)
        elif has_text and classification in ['text', 'mixed']:
            return ('text', Config.TEXT_QUEUE)
        elif classification == 'mixed':
            return ('ocr', Config.OCR_QUEUE)
        elif classification == 'blank':
            return ('text', Config.TEXT_QUEUE)  # Пустая страница
        else:
            # По умолчанию - OCR
            return ('ocr', Config.OCR_QUEUE)

    def process_pdf(self, pdf_path: str) -> str:
        """
        Обработать PDF файл - классифицировать страницы и поставить в очередь

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            job_id: Идентификатор задачи
        """
        logger.info(f"Начало обработки PDF: {pdf_path}")

        try:
            # Открыть PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)

            logger.info(f"PDF содержит {total_pages} страниц")

            # Создать задачу
            job_id = self.tracker.create_job(pdf_path, total_pages)
            logger.info(f"Создана задача: {job_id}")

            # Обработать каждую страницу
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]

                    # Определить тип страницы
                    page_type, queue_name = self.determine_page_type(page)

                    logger.info(
                        f"Страница {page_num + 1}/{total_pages}: "
                        f"тип={page_type}, очередь={queue_name}"
                    )

                    # Выбрать очередь
                    queue = self.text_queue if page_type == 'text' else self.ocr_queue

                    # Поставить задачу в очередь
                    if page_type == 'text':
                        from .workers.text_worker import process_text_page
                        job = queue.enqueue(
                            process_text_page,
                            args=(pdf_path, page_num, job_id),
                            retry=Retry(max=Config.MAX_RETRIES, interval=Config.RETRY_DELAY),
                            job_timeout=Config.TASK_TIMEOUT
                        )
                    else:
                        from .workers.ocr_worker import process_ocr_page
                        job = queue.enqueue(
                            process_ocr_page,
                            args=(pdf_path, page_num, job_id),
                            retry=Retry(max=Config.MAX_RETRIES, interval=Config.RETRY_DELAY),
                            job_timeout=Config.TASK_TIMEOUT
                        )

                    # Зарегистрировать задачу страницы
                    self.tracker.register_page_task(
                        job_id, page_num, page_type, job.id
                    )

                except Exception as e:
                    logger.error(
                        f"Ошибка обработки страницы {page_num}: {e}",
                        exc_info=True
                    )
                    # Отметить страницу как неудачную
                    self.tracker.mark_page_completed(
                        job_id, page_num, "", success=False, error=str(e)
                    )

            doc.close()

            logger.info(
                f"Задача {job_id}: все страницы поставлены в очередь"
            )
            return job_id

        except Exception as e:
            logger.error(f"Критическая ошибка обработки PDF: {e}", exc_info=True)
            raise
