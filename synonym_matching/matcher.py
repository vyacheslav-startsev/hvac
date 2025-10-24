"""
Гибридный matcher для сопоставления оборудования с синонимами
Использует Qdrant для semantic matching
Реализует трехуровневый подход: exact → fuzzy → semantic (Qdrant)
"""
import json
import logging
import os
from typing import Optional, Dict, List, Any
from rapidfuzz import fuzz, process
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EquipmentMatcher:
    """
    Гибридный matcher для оборудования с Qdrant

    Использует трехуровневую стратегию:
    1. Exact match - точное совпадение (мгновенно)
    2. Fuzzy match - нечеткое совпадение для опечаток (быстро, RapidFuzz)
    3. Semantic match - семантическое совпадение через Qdrant vector search
    """

    def __init__(
        self,
        synonym_db_path: str = "equipment_synonyms.json",
        qdrant_host: str = None,
        qdrant_port: int = None,
        collection_name: str = None
    ):
        """
        Инициализация matcher

        Args:
            synonym_db_path: Путь к JSON файлу с синонимами
            qdrant_host: Qdrant host (по умолчанию из env)
            qdrant_port: Qdrant port (по умолчанию из env)
            collection_name: Название коллекции (по умолчанию из env)
        """
        self.synonym_db_path = synonym_db_path
        self.synonym_db = self._load_synonyms()
        self.all_synonyms = self._build_synonym_list()

        # Qdrant настройки
        self.qdrant_host = qdrant_host or os.getenv('QDRANT_HOST', 'localhost')
        self.qdrant_port = qdrant_port or int(os.getenv('QDRANT_PORT', '6333'))
        self.collection_name = collection_name or os.getenv('QDRANT_COLLECTION_NAME', 'equipment_synonyms')

        # Lazy initialization
        self._qdrant_client = None
        self._semantic_model = None

        logger.info(
            f"Загружено {len(self.synonym_db.get('equipment_synonyms', []))} "
            f"категорий оборудования"
        )
        logger.info(f"Qdrant: {self.qdrant_host}:{self.qdrant_port}, коллекция: {self.collection_name}")

    def _load_synonyms(self) -> Dict:
        """Загрузить базу синонимов из JSON"""
        try:
            with open(self.synonym_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Файл синонимов не найден: {self.synonym_db_path}")
            return {"equipment_synonyms": []}
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}")
            return {"equipment_synonyms": []}

    def _build_synonym_list(self) -> List[tuple]:
        """
        Построить flat список всех синонимов для exact/fuzzy matching

        Returns:
            List[(synonym, canonical, category, language)]
        """
        synonyms_list = []

        for entry in self.synonym_db.get('equipment_synonyms', []):
            canonical = entry['canonical']
            category = entry.get('category', 'unknown')

            # Добавить английские синонимы
            for syn in entry.get('synonyms_en', []):
                synonyms_list.append((syn, canonical, category, 'en'))

            # Добавить русские синонимы
            for syn in entry.get('synonyms_ru', []):
                synonyms_list.append((syn, canonical, category, 'ru'))

            # Добавить аббревиатуры из metadata
            if 'metadata' in entry and 'abbreviations' in entry['metadata']:
                for abbr in entry['metadata']['abbreviations']:
                    synonyms_list.append((abbr, canonical, category, 'abbr'))

        return synonyms_list

    def _get_qdrant_client(self):
        """Lazy initialization Qdrant client"""
        if self._qdrant_client is None:
            try:
                logger.info(f"Подключение к Qdrant {self.qdrant_host}:{self.qdrant_port}...")
                self._qdrant_client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port
                )

                # Проверить существование коллекции
                collections = self._qdrant_client.get_collections()
                if not any(col.name == self.collection_name for col in collections.collections):
                    logger.warning(
                        f"Коллекция '{self.collection_name}' не найдена в Qdrant. "
                        f"Semantic search не будет работать. "
                        f"Запустите: python init_qdrant.py"
                    )
                    self._qdrant_client = None
                else:
                    logger.info(f"✓ Qdrant подключен, коллекция: {self.collection_name}")

            except Exception as e:
                logger.error(f"Ошибка подключения к Qdrant: {e}")
                logger.warning("Semantic matching через Qdrant не доступен")
                self._qdrant_client = None

        return self._qdrant_client

    def _get_semantic_model(self):
        """Lazy loading semantic model для создания query embeddings"""
        if self._semantic_model is None:
            try:
                model_name = os.getenv('SEMANTIC_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')
                logger.info(f"Загрузка semantic model: {model_name}...")
                self._semantic_model = SentenceTransformer(model_name)
                logger.info("✓ Semantic model загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки semantic model: {e}")
                self._semantic_model = None

        return self._semantic_model

    def match(
        self,
        query: str,
        method: str = 'hybrid',
        confidence_threshold: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        Сопоставить запрос с базой синонимов

        Args:
            query: Название оборудования для поиска
            method: 'exact', 'fuzzy', 'semantic', 'hybrid'
            confidence_threshold: Минимальная уверенность (0.0-1.0)

        Returns:
            Dict с информацией о совпадении или None
        """
        if not query or not query.strip():
            return None

        query = query.strip()

        # Tier 1: Exact match
        exact_result = self._exact_match(query)
        if exact_result:
            logger.debug(f"Exact match: '{query}' -> {exact_result['canonical']}")
            return exact_result

        # Tier 2: Fuzzy match
        if method in ['fuzzy', 'hybrid']:
            fuzzy_result = self._fuzzy_match(query, threshold=85)
            if fuzzy_result and fuzzy_result['confidence'] >= confidence_threshold:
                logger.debug(
                    f"Fuzzy match: '{query}' -> {fuzzy_result['canonical']} "
                    f"({fuzzy_result['confidence']:.2f})"
                )
                return fuzzy_result

        # Tier 3: Semantic match через Qdrant
        if method in ['semantic', 'hybrid']:
            semantic_result = self._semantic_match_qdrant(query, threshold=confidence_threshold)
            if semantic_result:
                logger.debug(
                    f"Semantic match (Qdrant): '{query}' -> {semantic_result['canonical']} "
                    f"({semantic_result['confidence']:.2f})"
                )
                return semantic_result

        logger.debug(f"No match found for: '{query}'")
        return None

    def _exact_match(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Точное совпадение (с учетом регистра и пробелов)

        Returns:
            Dict с информацией о совпадении или None
        """
        query_norm = query.lower().strip()

        for syn, canonical, category, lang in self.all_synonyms:
            if syn.lower().strip() == query_norm:
                return {
                    'canonical': canonical,
                    'matched_text': syn,
                    'category': category,
                    'language': lang,
                    'method': 'exact',
                    'confidence': 1.0
                }

        return None

    def _fuzzy_match(self, query: str, threshold: int = 80) -> Optional[Dict[str, Any]]:
        """
        Нечеткое совпадение с использованием RapidFuzz

        Args:
            query: Запрос
            threshold: Минимальный score (0-100)

        Returns:
            Dict с информацией о совпадении или None
        """
        # Получить все синонимы для поиска
        synonym_texts = [s[0] for s in self.all_synonyms]

        # Найти лучшие совпадения
        matches = process.extract(
            query,
            synonym_texts,
            scorer=fuzz.WRatio,
            limit=3
        )

        if matches and matches[0][1] >= threshold:
            matched_syn = matches[0][0]
            score = matches[0][1]

            # Найти canonical для совпадения
            for syn, canonical, category, lang in self.all_synonyms:
                if syn == matched_syn:
                    return {
                        'canonical': canonical,
                        'matched_text': matched_syn,
                        'category': category,
                        'language': lang,
                        'method': 'fuzzy',
                        'confidence': score / 100.0
                    }

        return None

    def _semantic_match_qdrant(
        self,
        query: str,
        threshold: float = 0.7,
        top_k: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Семантическое совпадение через Qdrant vector search

        Args:
            query: Запрос
            threshold: Минимальная cosine similarity (0.0-1.0)
            top_k: Количество лучших результатов

        Returns:
            Dict с информацией о совпадении или None
        """
        # Получить Qdrant client
        client = self._get_qdrant_client()
        if client is None:
            logger.debug("Qdrant недоступен, пропуск semantic matching")
            return None

        # Получить semantic model для query embedding
        model = self._get_semantic_model()
        if model is None:
            logger.debug("Semantic model недоступна")
            return None

        try:
            # Создать embedding для query
            query_vector = model.encode([query])[0].tolist()

            # Поиск в Qdrant
            search_result = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=threshold
            )

            if search_result:
                best_match = search_result[0]

                return {
                    'canonical': best_match.payload['canonical'],
                    'matched_text': best_match.payload['synonym'],
                    'category': best_match.payload.get('category', 'unknown'),
                    'language': best_match.payload.get('language', 'unknown'),
                    'method': 'semantic_qdrant',
                    'confidence': float(best_match.score)
                }

        except Exception as e:
            logger.error(f"Ошибка Qdrant search: {e}")

        return None

    def batch_match(
        self,
        queries: List[str],
        method: str = 'hybrid',
        confidence_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Сопоставить несколько запросов

        Args:
            queries: Список названий оборудования
            method: Метод сопоставления
            confidence_threshold: Минимальная уверенность

        Returns:
            Список результатов (с None для несовпавших)
        """
        results = []

        for query in queries:
            result = self.match(query, method, confidence_threshold)
            results.append({
                'query': query,
                'match': result
            })

        return results

    def get_all_canonical_names(self) -> List[str]:
        """Получить список всех канонических названий"""
        return [
            entry['canonical']
            for entry in self.synonym_db.get('equipment_synonyms', [])
        ]

    def get_synonyms_for_canonical(self, canonical: str) -> Dict[str, List[str]]:
        """
        Получить все синонимы для канонического названия

        Args:
            canonical: Каноническое название

        Returns:
            Dict с ключами 'synonyms_en', 'synonyms_ru'
        """
        for entry in self.synonym_db.get('equipment_synonyms', []):
            if entry['canonical'] == canonical:
                return {
                    'synonyms_en': entry.get('synonyms_en', []),
                    'synonyms_ru': entry.get('synonyms_ru', []),
                    'category': entry.get('category', 'unknown')
                }

        return {'synonyms_en': [], 'synonyms_ru': [], 'category': 'unknown'}


# Пример использования
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Создать matcher с Qdrant
    matcher = EquipmentMatcher(synonym_db_path='../equipment_synonyms.json')

    # Тестовые запросы
    test_queries = [
        "центробежный насос",
        "центробешный насос",  # опечатка -> fuzzy
        "AHU",  # аббревиатура -> exact
        "помпа",  # семантическое -> qdrant
        "теплообменник",
        "chiller",
        "fan coil"
    ]

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ SYNONYM MATCHER (QDRANT)")
    print("=" * 80)

    for query in test_queries:
        result = matcher.match(query, method='hybrid', confidence_threshold=0.7)

        if result:
            print(f"\n✓ '{query}'")
            print(f"  → Canonical: {result['canonical']}")
            print(f"  → Matched: {result['matched_text']}")
            print(f"  → Method: {result['method']}")
            print(f"  → Confidence: {result['confidence']:.2f}")
            print(f"  → Category: {result['category']}")
        else:
            print(f"\n✗ '{query}' - не найдено совпадений")

    print("\n" + "=" * 80)
