"""
PDF Processing Package
Handles PDF text extraction via Redis queues
"""
import logging
from typing import Dict, Any
from redis import Redis

from config import Config
from .coordinator import PDFCoordinator
from .tracker import TaskTracker
from .assembler import ResultAssembler

logger = logging.getLogger(__name__)


class PDFProcessingModule:
    """
    Модуль обработки PDF через Redis очереди
    Использует text_worker и ocr_worker для распределенной обработки
    """

    def __init__(self, redis_config: Dict[str, Any]):
        """
        Args:
            redis_config: Redis connection configuration
        """
        self.redis_config = redis_config

    def process(self, pdf_path: str) -> Dict[str, Any]:
        """
        Загрузить и обработать PDF через Redis очереди

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with extracted text and metadata
        """
        logger.info(f"Загрузка PDF через Redis очереди: {pdf_path}")
        logger.info("ВАЖНО: Убедитесь что воркеры запущены:")
        logger.info("  python text_worker.py")
        logger.info("  python ocr_worker.py")

        # Подключиться к Redis
        redis_client = Redis(**self.redis_config)

        # Создать компоненты обработки
        coordinator = PDFCoordinator(redis_client)
        tracker = TaskTracker(redis_client)

        # Поставить PDF в очередь обработки
        logger.info("Анализ PDF и постановка страниц в очереди...")
        job_id = coordinator.process_pdf(pdf_path)

        logger.info(f"✓ Job ID: {job_id}")
        logger.info("✓ Страницы распределены по очередям (text/OCR)")
        logger.info("Ожидание обработки воркерами...")

        # Ждать завершения обработки воркерами
        timeout = Config.TASK_TIMEOUT
        completed = tracker.wait_for_completion(
            job_id,
            timeout=timeout,
            poll_interval=2
        )

        if not completed:
            raise TimeoutError(
                f"Превышен таймаут ожидания обработки ({timeout}s). "
                f"Проверьте что воркеры запущены и обрабатывают задачи."
            )

        # Получить результаты
        status = tracker.get_job_status(job_id)
        results = tracker.get_results(job_id)

        logger.info(
            f"✓ Redis обработка завершена: "
            f"{status['completed_pages']}/{status['total_pages']} страниц, "
            f"text={status.get('text_pages', 0)}, "
            f"OCR={status.get('ocr_pages', 0)}, "
            f"ошибок={status['failed_pages']}"
        )

        # Подготовить данные для следующего этапа
        page_texts = {}
        for page_num, text in results.items():
            page_texts[page_num] = text

        full_text = "\n\n".join([page_texts[i] for i in sorted(page_texts.keys())])

        return {
            "pdf_path": pdf_path,
            "full_text": full_text,
            "page_texts": page_texts,
            "total_pages": int(status['total_pages']),
            "job_id": job_id,
            "redis_stats": {
                "completed_pages": int(status['completed_pages']),
                "failed_pages": int(status['failed_pages']),
                "text_pages": int(status.get('text_pages', 0)),
                "ocr_pages": int(status.get('ocr_pages', 0))
            }
        }


__all__ = [
    'PDFProcessingModule',
    'PDFCoordinator',
    'TaskTracker',
    'ResultAssembler',
]
