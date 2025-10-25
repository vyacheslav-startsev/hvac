"""
Конфигурация приложения
"""
import os
from dotenv import load_dotenv

# Загрузить переменные окружения из .env файла
load_dotenv()


class Config:
    """Настройки приложения"""

    # Redis настройки
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None) or None

    # Очереди
    TEXT_QUEUE = 'text_page_queue'
    OCR_QUEUE = 'ocr_page_queue'

    # LangChain и LLM настройки
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

    # LLM модели
    PRIMARY_LLM_MODEL = os.getenv('PRIMARY_LLM_MODEL', 'gpt-4o')
    FALLBACK_LLM_MODEL = os.getenv('FALLBACK_LLM_MODEL', 'claude-3-5-sonnet-20241022')
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0'))

    # Synonym matching настройки
    SYNONYM_DB_PATH = os.getenv('SYNONYM_DB_PATH', 'equipment_synonyms.json')
    SYNONYM_MATCH_METHOD = os.getenv('SYNONYM_MATCH_METHOD', 'hybrid')
    SYNONYM_CONFIDENCE_THRESHOLD = float(os.getenv('SYNONYM_CONFIDENCE_THRESHOLD', '0.8'))

    # Semantic model
    SEMANTIC_MODEL_NAME = os.getenv('SEMANTIC_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')

    # Qdrant настройки
    QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
    QDRANT_HTTP_PORT = int(os.getenv('QDRANT_HTTP_PORT', '6333'))
    QDRANT_GRPC_PORT = int(os.getenv('QDRANT_GRPC_PORT', '6334'))
    QDRANT_COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'equipment_synonyms')

    # Extraction настройки
    EXTRACTION_MAX_TEXT_LENGTH = int(os.getenv('EXTRACTION_MAX_TEXT_LENGTH', '15000'))
    EXTRACTION_MAX_RETRIES = int(os.getenv('EXTRACTION_MAX_RETRIES', '3'))

    # PDF классификация
    IMAGE_COVERAGE_THRESHOLD = float(os.getenv('IMAGE_COVERAGE_THRESHOLD', '0.5'))
    TEXT_COVERAGE_THRESHOLD = float(os.getenv('TEXT_COVERAGE_THRESHOLD', '0.1'))

    # Worker настройки
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))
    TASK_TIMEOUT = int(os.getenv('TASK_TIMEOUT', '600'))

    # Директории
    TEMP_DIR = os.getenv('TEMP_DIR', './temp')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', './output')

    @staticmethod
    def ensure_dirs():
        """Создать необходимые директории"""
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
