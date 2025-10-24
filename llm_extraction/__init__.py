"""
LLM Extraction Package
Handles equipment extraction from text using LLM
"""
import logging
from typing import Dict, Any
from config import Config
from .extractor import EquipmentExtractor

logger = logging.getLogger(__name__)


class LLMExtractionModule:
    """
    Модуль извлечения оборудования через LLM
    Использует постраничную обработку для больших документов
    """

    def __init__(
        self,
        primary_model: str = None,
        fallback_model: str = None,
        temperature: float = None,
        max_retries: int = None
    ):
        """
        Args:
            primary_model: Primary LLM model name
            fallback_model: Fallback LLM model name
            temperature: LLM temperature
            max_retries: Maximum retry attempts
        """
        self.extractor = EquipmentExtractor(
            primary_model=primary_model or Config.PRIMARY_LLM_MODEL,
            fallback_model=fallback_model or Config.FALLBACK_LLM_MODEL,
            temperature=temperature or Config.LLM_TEMPERATURE,
            max_retries=max_retries or Config.EXTRACTION_MAX_RETRIES
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Извлечь оборудование из текста

        Args:
            data: Dict with page_texts from previous stage

        Returns:
            Dict with raw_equipment list added
        """
        logger.info("Извлечение данных об оборудовании через LLM...")

        # Постраничное извлечение
        equipment_list = self.extractor.extract_from_pages(
            data["page_texts"],
            max_length_per_page=Config.EXTRACTION_MAX_TEXT_LENGTH
        )

        if equipment_list is None:
            logger.error("Ошибка извлечения оборудования")
            equipment_list = []

        data["raw_equipment"] = equipment_list
        data["extraction_count"] = len(equipment_list)

        logger.info(f"✓ Извлечено {len(equipment_list)} единиц оборудования")

        return data


__all__ = [
    'LLMExtractionModule',
    'EquipmentExtractor',
]
