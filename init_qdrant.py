"""
Скрипт инициализации Qdrant векторной базы
Загружает синонимы из equipment_synonyms.json в Qdrant при первом запуске
"""
import json
import logging
import os
import sys
import time
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Настройки
QDRANT_HOST = os.getenv('QDRANT_HOST', 'localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', '6333'))
COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME', 'equipment_synonyms')
SYNONYM_DB_PATH = os.getenv('SYNONYM_DB_PATH', 'equipment_synonyms.json')
SEMANTIC_MODEL_NAME = os.getenv('SEMANTIC_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')


def wait_for_qdrant(host: str, port: int, max_retries: int = 30, retry_delay: int = 2):
    """
    Ждать доступности Qdrant

    Args:
        host: Qdrant host
        port: Qdrant port
        max_retries: Максимум попыток
        retry_delay: Задержка между попытками (сек)
    """
    logger.info(f"Ожидание доступности Qdrant {host}:{port}...")

    for i in range(max_retries):
        try:
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            logger.info("✓ Qdrant доступен")
            return True
        except Exception as e:
            logger.warning(f"Попытка {i+1}/{max_retries}: Qdrant недоступен ({e})")
            if i < max_retries - 1:
                time.sleep(retry_delay)

    raise ConnectionError(f"Qdrant недоступен после {max_retries} попыток")


def load_synonyms(file_path: str) -> Dict:
    """Загрузить синонимы из JSON"""
    logger.info(f"Загрузка синонимов из {file_path}...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        synonym_count = len(data.get('equipment_synonyms', []))
        logger.info(f"✓ Загружено {synonym_count} категорий оборудования")

        return data

    except FileNotFoundError:
        logger.error(f"Файл не найден: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        raise


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Проверить существование коллекции"""
    try:
        collections = client.get_collections()
        return any(col.name == collection_name for col in collections.collections)
    except Exception as e:
        logger.error(f"Ошибка проверки коллекции: {e}")
        return False


def init_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 384
):
    """
    Инициализировать коллекцию Qdrant

    Args:
        client: Qdrant client
        collection_name: Название коллекции
        vector_size: Размер вектора (384 для multilingual-MiniLM-L12-v2)
    """
    if collection_exists(client, collection_name):
        logger.info(f"Коллекция '{collection_name}' уже существует")

        # Проверить количество точек
        collection_info = client.get_collection(collection_name)
        points_count = collection_info.points_count

        if points_count > 0:
            logger.info(f"Коллекция содержит {points_count} точек")
            logger.info("Пропуск инициализации - данные уже загружены")
            return False
        else:
            logger.info("Коллекция пустая, продолжаем загрузку")
            return True
    else:
        logger.info(f"Создание новой коллекции '{collection_name}'...")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        logger.info(f"✓ Коллекция '{collection_name}' создана")
        return True


def index_synonyms(
    client: QdrantClient,
    collection_name: str,
    synonym_db: Dict,
    model: SentenceTransformer
):
    """
    Индексировать синонимы в Qdrant

    Args:
        client: Qdrant client
        collection_name: Название коллекции
        synonym_db: База синонимов
        model: Sentence transformer model
    """
    logger.info("Индексирование синонимов...")

    points = []
    point_id = 0

    for entry in synonym_db.get('equipment_synonyms', []):
        canonical = entry['canonical']
        category = entry.get('category', 'unknown')

        # Индексировать английские синонимы
        for syn in entry.get('synonyms_en', []):
            embedding = model.encode([syn])[0].tolist()

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "canonical": canonical,
                    "synonym": syn,
                    "language": "en",
                    "category": category
                }
            )
            points.append(point)
            point_id += 1

        # Индексировать русские синонимы
        for syn in entry.get('synonyms_ru', []):
            embedding = model.encode([syn])[0].tolist()

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "canonical": canonical,
                    "synonym": syn,
                    "language": "ru",
                    "category": category
                }
            )
            points.append(point)
            point_id += 1

        # Индексировать аббревиатуры
        if 'metadata' in entry and 'abbreviations' in entry['metadata']:
            for abbr in entry['metadata']['abbreviations']:
                embedding = model.encode([abbr])[0].tolist()

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "canonical": canonical,
                        "synonym": abbr,
                        "language": "abbr",
                        "category": category
                    }
                )
                points.append(point)
                point_id += 1

    # Загрузить все точки в Qdrant
    if points:
        logger.info(f"Загрузка {len(points)} точек в Qdrant...")

        # Загружать батчами для производительности
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            logger.info(f"Загружено {min(i+batch_size, len(points))}/{len(points)} точек")

        logger.info(f"✓ Все {len(points)} точек загружены в Qdrant")
    else:
        logger.warning("Нет точек для загрузки")


def main():
    """Главная функция инициализации"""
    logger.info("=" * 80)
    logger.info("ИНИЦИАЛИЗАЦИЯ QDRANT ВЕКТОРНОЙ БАЗЫ")
    logger.info("=" * 80)

    try:
        # Ждать доступности Qdrant
        wait_for_qdrant(QDRANT_HOST, QDRANT_PORT)

        # Подключиться к Qdrant
        logger.info(f"Подключение к Qdrant {QDRANT_HOST}:{QDRANT_PORT}...")
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info("✓ Подключение установлено")

        # Инициализировать коллекцию
        should_index = init_qdrant_collection(client, COLLECTION_NAME)

        if not should_index:
            logger.info("=" * 80)
            logger.info("Инициализация не требуется - данные уже загружены")
            logger.info("=" * 80)
            return 0

        # Загрузить синонимы из JSON
        synonym_db = load_synonyms(SYNONYM_DB_PATH)

        # Загрузить semantic model
        logger.info(f"Загрузка semantic model: {SEMANTIC_MODEL_NAME}...")
        model = SentenceTransformer(SEMANTIC_MODEL_NAME)
        logger.info("✓ Model загружена")

        # Индексировать синонимы
        index_synonyms(client, COLLECTION_NAME, synonym_db, model)

        # Проверить результат
        collection_info = client.get_collection(COLLECTION_NAME)
        logger.info("=" * 80)
        logger.info("ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
        logger.info("=" * 80)
        logger.info(f"Коллекция: {COLLECTION_NAME}")
        logger.info(f"Всего точек: {collection_info.points_count}")
        logger.info(f"Размер вектора: {collection_info.config.params.vectors.size}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error("=" * 80)
        logger.error("ОШИБКА ИНИЦИАЛИЗАЦИИ")
        logger.error("=" * 80)
        logger.error(f"Ошибка: {e}", exc_info=True)
        logger.error("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
