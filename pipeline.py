"""
Полный pipeline обработки PDF
Использует LangChain для оркестрации + Redis для распределенной обработки PDF

АРХИТЕКТУРА PIPELINE:
====================

┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Text Extraction через Redis Queues           │
│  - PDFCoordinator анализирует и классифицирует страницы │
│  - Текстовые → text_worker (PyMuPDF)                    │
│  - Графические → ocr_worker (PaddleOCR)                 │
│  - TaskTracker отслеживает прогресс                     │
│  Output: {pdf_path, page_texts, redis_stats, job_id}    │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: LLM Extraction                                │
│  - EquipmentExtractor с постраничной обработкой         │
│  - GPT-4o (primary) + Claude 3.5 Sonnet (fallback)     │
│  - Structured output через Pydantic schemas             │
│  Output: raw_equipment [name, model, quantity, etc.]    │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Synonym Matching                              │
│  - EquipmentMatcher с гибридным подходом                │
│  - Tier 1: Exact → Tier 2: Fuzzy → Tier 3: Semantic    │
│  Output: matched_equipment, unmatched_equipment         │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4: Output Formatting & Save                      │
│  - JSON с metadata, statistics, equipment               │
│  Output: Сохраненный JSON файл                          │
└─────────────────────────────────────────────────────────┘

ТРЕБОВАНИЯ:
===========
- Redis server должен быть запущен
- Воркеры должны быть запущены:
    python text_worker.py &
    python ocr_worker.py &
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain_core.runnables import RunnableLambda

# Локальные модули
from config import Config
from redis import Redis
from pdf_processing import PDFProcessingModule
from llm_extraction import LLMExtractionModule
from synonym_matching import SynonymMatchingModule
from result_formatting import ResultFormattingModule

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EquipmentProcessingPipeline:
    """
    Pipeline с использованием Redis очередей для PDF обработки
    """

    def __init__(
        self,
        synonym_db_path: str = None,
        output_dir: str = "./output",
        redis_host: str = None,
        redis_port: int = None,
        redis_db: int = None,
        redis_password: str = None
    ):
        """
        Инициализация pipeline

        Args:
            synonym_db_path: Путь к базе синонимов
            output_dir: Директория для сохранения результатов
            redis_host: Redis host (по умолчанию из Config)
            redis_port: Redis port (по умолчанию из Config)
            redis_db: Redis database (по умолчанию из Config)
            redis_password: Redis password (по умолчанию из Config)
        """
        self.synonym_db_path = synonym_db_path or Config.SYNONYM_DB_PATH
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Redis настройки
        self.redis_config = {
            'host': redis_host or Config.REDIS_HOST,
            'port': redis_port or Config.REDIS_PORT,
            'db': redis_db or Config.REDIS_DB,
            'password': redis_password or Config.REDIS_PASSWORD,
            'decode_responses': False
        }

        # Инициализировать компоненты
        logger.info("Инициализация компонентов pipeline...")

        # Проверить Redis подключение
        self._check_redis_connection()

        # Инициализировать модули
        self.pdf_module = PDFProcessingModule(self.redis_config)
        self.llm_module = LLMExtractionModule()
        self.synonym_module = SynonymMatchingModule(synonym_db_path=self.synonym_db_path)
        self.result_module = ResultFormattingModule()

        # Создать LangChain pipeline
        self.pipeline = self._build_pipeline()

        logger.info("Pipeline инициализирован")

    def _check_redis_connection(self):
        """Проверить подключение к Redis"""
        try:
            redis_client = Redis(**self.redis_config)
            redis_client.ping()
            logger.info("✓ Redis подключение успешно")
        except Exception as e:
            logger.error(f"✗ Ошибка подключения к Redis: {e}")
            raise ConnectionError(
                "Redis недоступен. Убедитесь что Redis server запущен.\n"
                f"Проверьте настройки: {self.redis_config['host']}:{self.redis_config['port']}"
            )

    def _build_pipeline(self):
        """
        Построить LangChain LCEL pipeline с модулями

        Pipeline: PDF → Redis Queues (Text/OCR) → LLM Extraction → Synonym Matching → JSON
        """
        # Собрать LCEL pipeline из модулей
        pipeline = (
            RunnableLambda(self.pdf_module.process)
            | RunnableLambda(self.llm_module.process)
            | RunnableLambda(self.synonym_module.process)
            | RunnableLambda(self.result_module.process)
        )

        return pipeline

    def process(
        self,
        pdf_path: str,
        output_filename: str = None
    ) -> Dict[str, Any]:
        """
        Обработать PDF файл через весь pipeline

        Args:
            pdf_path: Путь к PDF файлу
            output_filename: Имя выходного JSON файла (опционально)

        Returns:
            Dict с результатами обработки
        """
        logger.info("=" * 80)
        logger.info(f"НАЧАЛО ОБРАБОТКИ (REDIS MODE): {pdf_path}")
        logger.info("=" * 80)

        try:
            # Выполнить pipeline
            result = self.pipeline.invoke(pdf_path)

            # Определить имя выходного файла
            if output_filename is None:
                pdf_name = Path(pdf_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{pdf_name}_equipment_{timestamp}.json"

            # Сохранить результат
            output_path = self.output_dir / output_filename
            self._save_results(result, output_path)

            # Вывести итоговую статистику
            self._print_summary(result, output_path)

            return result

        except Exception as e:
            logger.error(f"Ошибка обработки PDF: {e}", exc_info=True)
            raise

    def _save_results(self, result: Dict[str, Any], output_path: Path):
        """Сохранить результаты в JSON файл"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ Результат сохранен: {output_path}")

    def _print_summary(self, result: Dict[str, Any], output_path: Path):
        """Вывести итоговую статистику"""
        logger.info("=" * 80)
        logger.info("ОБРАБОТКА ЗАВЕРШЕНА")
        logger.info("=" * 80)
        logger.info(f"Результат: {output_path}")

        # Redis статистика
        if result.get("redis_statistics"):
            redis_stats = result["redis_statistics"]
            logger.info(f"\nRedis обработка:")
            logger.info(f"  Всего страниц: {redis_stats.get('completed_pages', 0)}")
            logger.info(f"  Текстовых: {redis_stats.get('text_pages', 0)}")
            logger.info(f"  OCR: {redis_stats.get('ocr_pages', 0)}")
            logger.info(f"  Ошибок: {redis_stats.get('failed_pages', 0)}")

        # Extraction статистика
        extr_stats = result["extraction_statistics"]
        logger.info(f"\nИзвлечение оборудования:")
        logger.info(f"  Найдено: {extr_stats['total']}")
        logger.info(f"  Сопоставлено: {extr_stats['matched']} ({extr_stats['match_rate']*100:.1f}%)")
        logger.info(f"  Требует проверки: {extr_stats['unmatched']}")
        logger.info("=" * 80)

    def generate_report(self, result: Dict[str, Any]) -> str:
        """
        Сгенерировать текстовый отчет

        Args:
            result: Результат обработки

        Returns:
            Текстовый отчет
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ОТЧЕТ ОБ ИЗВЛЕЧЕННОМ ОБОРУДОВАНИИ (REDIS MODE)")
        lines.append("=" * 80)
        lines.append(f"\nИсходный файл: {result['metadata']['source_pdf']}")
        lines.append(f"Дата обработки: {result['metadata']['processed_at']}")
        lines.append(f"Redis Job ID: {result['metadata'].get('redis_job_id', 'N/A')}")

        # Redis статистика
        if result.get("redis_statistics"):
            redis_stats = result["redis_statistics"]
            lines.append(f"\nRedis обработка:")
            lines.append(f"  Всего страниц: {redis_stats.get('completed_pages', 0)}")
            lines.append(f"  Текстовых страниц: {redis_stats.get('text_pages', 0)}")
            lines.append(f"  OCR страниц: {redis_stats.get('ocr_pages', 0)}")
            lines.append(f"  Ошибок обработки: {redis_stats.get('failed_pages', 0)}")

        stats = result['extraction_statistics']
        lines.append(f"\nИзвлечение и сопоставление:")
        lines.append(f"  Всего найдено: {stats['total']}")
        lines.append(f"  Сопоставлено: {stats['matched']} ({stats['match_rate']*100:.1f}%)")
        lines.append(f"  Не сопоставлено: {stats['unmatched']}")

        # Сопоставленное оборудование
        if result['equipment']['matched']:
            lines.append(f"\n{'=' * 80}")
            lines.append("СОПОСТАВЛЕННОЕ ОБОРУДОВАНИЕ:")
            lines.append("=" * 80)

            for i, item in enumerate(result['equipment']['matched'], 1):
                lines.append(f"\n{i}. {item['original_name']}")
                lines.append(f"   → Каноническое название: {item['canonical_name']}")
                lines.append(f"   → Категория: {item.get('category', 'N/A')}")
                lines.append(f"   → Метод: {item['match_method']}, Уверенность: {item['match_confidence']:.2%}")
                if item.get('model'):
                    lines.append(f"   → Модель: {item['model']}")
                if item.get('quantity'):
                    lines.append(f"   → Количество: {item['quantity']}")
                if item.get('source_page') is not None:
                    lines.append(f"   → Страница: {item['source_page'] + 1}")

        # Несопоставленное оборудование
        if result['equipment']['unmatched']:
            lines.append(f"\n{'=' * 80}")
            lines.append("НЕСОПОСТАВЛЕННОЕ ОБОРУДОВАНИЕ (требует проверки):")
            lines.append("=" * 80)

            for i, item in enumerate(result['equipment']['unmatched'], 1):
                lines.append(f"\n{i}. {item['original_name']}")
                if item.get('model'):
                    lines.append(f"   → Модель: {item['model']}")
                if item.get('quantity'):
                    lines.append(f"   → Количество: {item['quantity']}")
                if item.get('source_page') is not None:
                    lines.append(f"   → Страница: {item['source_page'] + 1}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


def main():
    """Пример использования pipeline с Redis"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Pipeline обработки PDF с Redis очередями для извлечения оборудования'
    )
    parser.add_argument('pdf_path', help='Путь к PDF файлу')
    parser.add_argument('--output', help='Имя выходного JSON файла')
    parser.add_argument('--output-dir', default='./output', help='Директория для результатов')
    parser.add_argument('--report', action='store_true', help='Вывести текстовый отчет')
    parser.add_argument('--synonym-db', help='Путь к файлу с синонимами')

    args = parser.parse_args()

    try:
        # Создать pipeline
        pipeline = EquipmentProcessingPipeline(
            synonym_db_path=args.synonym_db,
            output_dir=args.output_dir
        )

        # Обработать PDF
        result = pipeline.process(
            pdf_path=args.pdf_path,
            output_filename=args.output
        )

        # Вывести отчет если запрошено
        if args.report:
            report = pipeline.generate_report(result)
            print("\n" + report)

    except ConnectionError as e:
        logger.error(f"\n{e}")
        logger.error("\nУбедитесь что:")
        logger.error("  1. Redis server запущен")
        logger.error("  2. Воркеры запущены:")
        logger.error("       python text_worker.py")
        logger.error("       python ocr_worker.py")
        exit(1)
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()
