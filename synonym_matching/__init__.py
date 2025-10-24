"""
Synonym Matching Package
Handles equipment name matching with synonym database
"""
import logging
from typing import Dict, Any
from config import Config
from .matcher import EquipmentMatcher

logger = logging.getLogger(__name__)


class SynonymMatchingModule:
    """
    Модуль сопоставления оборудования с базой синонимов
    Использует гибридный подход: exact → fuzzy → semantic
    """

    def __init__(
        self,
        synonym_db_path: str = None,
        match_method: str = None,
        confidence_threshold: float = None
    ):
        """
        Args:
            synonym_db_path: Path to synonym database
            match_method: Matching method (exact, fuzzy, semantic, hybrid)
            confidence_threshold: Minimum confidence threshold
        """
        self.matcher = EquipmentMatcher(
            synonym_db_path=synonym_db_path or Config.SYNONYM_DB_PATH
        )
        self.match_method = match_method or Config.SYNONYM_MATCH_METHOD
        self.confidence_threshold = confidence_threshold or Config.SYNONYM_CONFIDENCE_THRESHOLD

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Сопоставить оборудование с базой синонимов

        Args:
            data: Dict with raw_equipment from previous stage

        Returns:
            Dict with matched_equipment and unmatched_equipment added
        """
        logger.info("Сопоставление с базой синонимов...")

        matched_equipment = []
        unmatched_equipment = []

        for equipment in data["raw_equipment"]:
            name = equipment.get("name", "")
            if not name:
                continue

            # Попытка сопоставления
            match_result = self.matcher.match(
                name,
                method=self.match_method,
                confidence_threshold=self.confidence_threshold
            )

            # Создать обогащенную запись
            enriched_item = {
                **equipment,
                "original_name": name
            }

            if match_result:
                enriched_item.update({
                    "canonical_name": match_result["canonical"],
                    "matched_synonym": match_result["matched_text"],
                    "match_method": match_result["method"],
                    "match_confidence": match_result["confidence"],
                    "category": match_result.get("category"),
                    "language": match_result.get("language"),
                    "match_status": "matched"
                })
                matched_equipment.append(enriched_item)
            else:
                enriched_item.update({
                    "canonical_name": None,
                    "match_status": "unmatched",
                    "requires_review": True
                })
                unmatched_equipment.append(enriched_item)

        data["matched_equipment"] = matched_equipment
        data["unmatched_equipment"] = unmatched_equipment
        data["match_statistics"] = {
            "total": len(data["raw_equipment"]),
            "matched": len(matched_equipment),
            "unmatched": len(unmatched_equipment),
            "match_rate": len(matched_equipment) / len(data["raw_equipment"]) if data["raw_equipment"] else 0
        }

        logger.info(
            f"✓ Сопоставление: {len(matched_equipment)}/{len(data['raw_equipment'])} "
            f"({data['match_statistics']['match_rate']*100:.1f}%)"
        )

        return data


__all__ = [
    'SynonymMatchingModule',
    'EquipmentMatcher',
]
