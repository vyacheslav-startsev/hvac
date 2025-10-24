"""
Result Formatting Package
Handles final result formatting and JSON structure
"""
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultFormattingModule:
    """
    Модуль форматирования финального результата
    Создает структурированный JSON с метаданными и статистикой
    """

    def __init__(self, pipeline_version: str = "2.0.0-redis"):
        """
        Args:
            pipeline_version: Pipeline version string
        """
        self.pipeline_version = pipeline_version

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Подготовить финальный результат

        Args:
            data: Dict with all processing results

        Returns:
            Formatted result dict ready for JSON serialization
        """
        logger.info("Форматирование результата...")

        result = {
            "metadata": {
                "source_pdf": data["pdf_path"],
                "processed_at": datetime.now().isoformat(),
                "total_pages": data["total_pages"],
                "pipeline_version": self.pipeline_version,
                "redis_job_id": data.get("job_id"),
                "processing_method": "redis_queues"
            },
            "redis_statistics": data.get("redis_stats", {}),
            "extraction_statistics": data["match_statistics"],
            "equipment": {
                "matched": data["matched_equipment"],
                "unmatched": data["unmatched_equipment"]
            }
        }

        return result


__all__ = ['ResultFormattingModule']
