# HVAC Equipment Extraction System

Автоматизированная система извлечения данных об оборудовании из PDF-документов с использованием OCR, LLM и векторного поиска.

## Описание системы

Система представляет собой распределенный пайплайн обработки PDF-документов, который:

1. **Извлекает текст** - анализирует страницы PDF и извлекает текст напрямую (для текстовых страниц) или через OCR (для изображений)
2. **Распознает оборудование** - использует LLM (GPT-4o/Claude) для структурированного извлечения данных об оборудовании (название, модель, количество)
3. **Сопоставляет синонимы** - нормализует названия оборудования через гибридный поиск (точное совпадение → fuzzy match → семантический поиск по векторным эмбеддингам)
4. **Формирует результат** - сохраняет структурированный JSON с метаданными, статистикой и списком оборудования

### Архитектура

- **Redis (Valkey)** - очереди задач для распределенной обработки страниц
- **Qdrant** - векторная база данных для семантического поиска синонимов
- **RQ Workers** - воркеры для параллельной обработки (text_worker для текстовых страниц, ocr_worker для графических). Позволяют **значительно сократить время обработки** за счет распределения нагрузки между несколькими процессами
- **PaddleOCR** - современная VLM на базе трансформеров, обеспечивает высокое качество распознавания за меньшее время, работает локально без внешних API
- **Coordinator** - координатор пайплайна, управляет всеми этапами обработки

### Workflow

```mermaid
graph TD
    A[PDF Document] --> B[Stage 1: Text Extraction]

    B --> B1[PDFCoordinator<br/>анализирует и классифицирует страницы]
    B1 --> B2[Текстовые страницы]
    B1 --> B3[Графические страницы]

    B2 --> B4[text_worker<br/>PyMuPDF]
    B3 --> B5[ocr_worker<br/>PaddleOCR]

    B4 --> B6[TaskTracker<br/>отслеживает прогресс]
    B5 --> B6

    B6 --> B7[Output: pdf_path, page_texts,<br/>redis_stats, job_id]

    B7 --> C[Stage 2: LLM Extraction]

    C --> C1[EquipmentExtractor<br/>постраничная обработка]
    C1 --> C2[GPT-4o primary]
    C1 --> C3[Claude 3.5 Sonnet fallback]

    C2 --> C4[Structured output<br/>через Pydantic schemas]
    C3 --> C4

    C4 --> C5[Output: raw_equipment<br/>name, model, quantity]

    C5 --> D[Stage 3: Synonym Matching]

    D --> D1[EquipmentMatcher<br/>гибридный подход]
    D1 --> D2[Tier 1: Exact Match]
    D2 --> D3[Tier 2: Fuzzy Match]
    D3 --> D4[Tier 3: Semantic Match]

    D4 --> D5[Output: matched_equipment,<br/>unmatched_equipment]

    D5 --> E[Stage 4: Output Formatting & Save]

    E --> E1[JSON с metadata,<br/>statistics, equipment]
    E1 --> E2[Сохраненный JSON файл]

    style B fill:#e1f5ff
    style C fill:#fff4e1
    style D fill:#ffe1f5
    style E fill:#e1ffe1
```



# Запуск системы в Docker

## Предварительные требования
- Docker и Docker Compose установлены
- PDF файлы находятся в папке `./pdfs`

## Быстрый старт

### 1. Настройка окружения
```bash
# Скопируйте .env.example в .env
cp .env.example .env

# Отредактируйте .env и укажите ваши API ключи:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY
```

### 2. Запуск системы
```bash
docker compose up -d
```

Это запустит:
- Redis (Valkey) - очередь задач
- Qdrant - векторная база для синонимов
- qdrant-init - инициализация синонимов (запустится автоматически после готовности Qdrant)
- 2 воркера для обработки текстовых страниц
- 2 воркера для OCR-обработки

### 3. Запуск обработки PDF

```bash
docker compose run --rm coordinator python run_pipeline.py pdfs/your_file.pdf
```

Эта команда запустит пайплайн обработки указанного PDF файла. Воркеры автоматически подхватят задачи из очередей и обработают страницы.

**Результат**: JSON файл с извлеченным оборудованием сохраняется в `./output/{pdf_name}_equipment_{timestamp}.json`

## Мониторинг

Просмотр логов воркеров:
```bash
docker compose logs -f text-worker ocr-worker
```

Проверка статуса очереди (подключитесь к Redis):
```bash
docker exec -it pdf_valkey valkey-cli
> LLEN text_page_queue
> LLEN ocr_page_queue
```

## Остановка системы

```bash
docker compose down
```

Для удаления данных (Redis и Qdrant):
```bash
docker compose down -v
```

## Структура volumes

- `./pdfs` → `/app/pdfs` (read-only) - входные PDF файлы
- `./output` → `/app/output` - результаты обработки (JSON файлы)
- `./temp` → `/app/temp` - временные файлы
- `./data/redis` → `/data` - данные Redis/Valkey
- `./data/qdrant` → `/qdrant/storage` - векторная база Qdrant
