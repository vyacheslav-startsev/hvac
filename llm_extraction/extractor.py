"""
Извлечение данных об оборудовании из текста с использованием LangChain и LLM
"""
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os

logger = logging.getLogger(__name__)


# Pydantic схемы для structured extraction
class Equipment(BaseModel):
    """Модель данных оборудования"""
    name: str = Field(description="Название оборудования")
    model: Optional[str] = Field(default=None, description="Модель или тип оборудования")
    specifications: Optional[str] = Field(default=None, description="Технические характеристики")
    quantity: Optional[int] = Field(default=None, description="Количество единиц")
    location: Optional[str] = Field(default=None, description="Местоположение или зона установки")


class DocumentData(BaseModel):
    """Структура извлеченных данных из документа"""
    equipment_list: List[Equipment] = Field(description="Список обнаруженного оборудования")


class EquipmentExtractor:
    """
    Извлечение структурированных данных об оборудовании из текста

    Использует LangChain LCEL с:
    - GPT-4o как primary модель (structured output, vision capable)
    - Claude 3.5 Sonnet как fallback модель (для длинных документов)
    - Автоматический retry при ошибках
    - Structured output через with_structured_output()
    """

    def __init__(
        self,
        primary_model: str = "gpt-4o",
        fallback_model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0,
        max_retries: int = 3
    ):
        """
        Инициализация extractor

        Args:
            primary_model: Основная LLM модель
            fallback_model: Резервная модель при ошибках
            temperature: Температура генерации (0 = детерминированно)
            max_retries: Максимальное количество попыток
        """
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self.max_retries = max_retries

        # Проверить наличие API ключей
        self._check_api_keys()

        # Создать цепочку извлечения
        self._extraction_chain = self._build_extraction_chain()

        logger.info(
            f"EquipmentExtractor инициализирован: "
            f"primary={primary_model}, fallback={fallback_model}"
        )

    def _check_api_keys(self):
        """Проверить наличие необходимых API ключей"""
        if "gpt" in self.primary_model.lower() and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY не установлен - GPT модели не будут работать")

        if "claude" in self.fallback_model.lower() and not os.getenv("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY не установлен - Claude модели не будут работать")

    def _build_extraction_chain(self):
        """
        Построить LCEL цепочку для извлечения данных

        Returns:
            Runnable chain с автоматическим fallback и retry
        """
        # Создать prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Вы эксперт по извлечению данных об оборудовании из технических документов. "
             "Извлекайте только информацию, явно указанную в тексте. "
             "Если информация отсутствует, не придумывайте её. "
             "Для русских и английских терминов оборудования используйте оригинальное написание из текста. "
             "Если количество не указано явно, оставьте поле quantity пустым."),
            ("human",
             "Извлеките список оборудования из следующего текста документа:\n\n{text}")
        ])

        # Primary LLM с structured output
        primary_llm = None
        if "gpt" in self.primary_model.lower() and os.getenv("OPENAI_API_KEY"):
            primary_llm = ChatOpenAI(
                model=self.primary_model,
                temperature=self.temperature
            ).with_structured_output(DocumentData)

        # Fallback LLM с structured output
        fallback_llm = None
        if "claude" in self.fallback_model.lower() and os.getenv("ANTHROPIC_API_KEY"):
            fallback_llm = ChatAnthropic(
                model=self.fallback_model,
                temperature=self.temperature
            ).with_structured_output(DocumentData)

        # Построить цепочку с fallback и retry
        if primary_llm and fallback_llm:
            chain = (prompt | primary_llm).with_fallbacks([
                prompt | fallback_llm
            ]).with_retry(stop_after_attempt=self.max_retries)
        elif primary_llm:
            chain = (prompt | primary_llm).with_retry(stop_after_attempt=self.max_retries)
        elif fallback_llm:
            chain = (prompt | fallback_llm).with_retry(stop_after_attempt=self.max_retries)
        else:
            logger.error("Ни одна LLM модель не доступна - проверьте API ключи")
            chain = None

        return chain

    def extract_from_text(
        self,
        text: str,
        max_length: int = 15000
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Извлечь оборудование из текста

        Args:
            text: Текст документа
            max_length: Максимальная длина текста для обработки

        Returns:
            Список словарей с данными оборудования или None при ошибке
        """
        if not text or not text.strip():
            logger.warning("Пустой текст для извлечения")
            return []

        if not self._extraction_chain:
            logger.error("Extraction chain не инициализирована")
            return None

        # Ограничить длину текста
        if len(text) > max_length:
            logger.warning(
                f"Текст обрезан с {len(text)} до {max_length} символов"
            )
            text = text[:max_length]

        try:
            logger.info(f"Извлечение оборудования из текста ({len(text)} символов)...")

            # Выполнить извлечение
            result = self._extraction_chain.invoke({"text": text})

            # Конвертировать в словари
            equipment_list = [e.model_dump() for e in result.equipment_list]

            logger.info(
                f"✓ Извлечено {len(equipment_list)} единиц оборудования"
            )

            return equipment_list

        except Exception as e:
            logger.error(f"Ошибка извлечения оборудования: {e}", exc_info=True)
            return None

    def extract_from_pages(
        self,
        page_texts: Dict[int, str],
        max_length_per_page: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Извлечь оборудование из нескольких страниц

        Args:
            page_texts: Dict {page_num: text}
            max_length_per_page: Макс длина текста на страницу

        Returns:
            Объединенный список оборудования со всех страниц
        """
        all_equipment = []
        seen_items = set()  # Для дедупликации

        for page_num, text in sorted(page_texts.items()):
            logger.info(f"Обработка страницы {page_num}...")

            equipment_list = self.extract_from_text(text, max_length_per_page)

            if equipment_list:
                # Дедупликация по имени оборудования
                for item in equipment_list:
                    item_key = (item['name'], item.get('model', ''))

                    if item_key not in seen_items:
                        item['source_page'] = page_num  # Добавить номер страницы
                        all_equipment.append(item)
                        seen_items.add(item_key)

        logger.info(
            f"Всего извлечено {len(all_equipment)} уникальных единиц "
            f"оборудования с {len(page_texts)} страниц"
        )

        return all_equipment


# Пример использования
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Для теста нужны API ключи в .env:
    # OPENAI_API_KEY=sk-...
    # ANTHROPIC_API_KEY=sk-ant-...

    # Создать extractor
    extractor = EquipmentExtractor()

    # Тестовый текст
    sample_text = """
    ТЕХНИЧЕСКАЯ СПЕЦИФИКАЦИЯ СИСТЕМЫ ВЕНТИЛЯЦИИ

    1. Приточно-вытяжная установка (ПВУ)
       - Модель: Systemair Topvex SR04
       - Производительность: 2500 м³/ч
       - Количество: 2 шт
       - Размещение: Машинное отделение, 5 этаж

    2. Центробежные насосы
       - Модель: Grundfos MAGNA3 50-120
       - Мощность: 0.5 кВт
       - Количество: 4 шт

    3. Теплообменник пластинчатый
       - Тип: GEA WP 700
       - Площадь теплообмена: 35 м²

    4. Система фильтрации:
       - Фильтры воздушные класса F7 - 8 шт
       - Сетчатые фильтры DN50 - 12 шт
    """

    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ EQUIPMENT EXTRACTOR")
    print("=" * 80)

    result = extractor.extract_from_text(sample_text)

    if result:
        print(f"\n✓ Найдено единиц оборудования: {len(result)}\n")

        for i, equipment in enumerate(result, 1):
            print(f"{i}. {equipment['name']}")
            if equipment.get('model'):
                print(f"   Модель: {equipment['model']}")
            if equipment.get('quantity'):
                print(f"   Количество: {equipment['quantity']} шт")
            if equipment.get('specifications'):
                print(f"   Характеристики: {equipment['specifications']}")
            if equipment.get('location'):
                print(f"   Размещение: {equipment['location']}")
            print()
    else:
        print("\n✗ Ошибка извлечения данных")
        print("Проверьте наличие API ключей в переменных окружения:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")

    print("=" * 80)
