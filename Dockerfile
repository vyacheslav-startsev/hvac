FROM python:3.11-slim

# Установить системные зависимости
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglx0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Создать рабочую директорию
WORKDIR /app

# Скопировать requirements и установить зависимости
COPY requirements.txt requirements-base.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать код приложения
COPY . .

# Предзагрузить модель SentenceTransformer
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Создать директорию для временных файлов
RUN mkdir -p /app/temp

# Переменные окружения по умолчанию
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_DB=0
ENV TEMP_DIR=/app/temp
ENV PYTHONUNBUFFERED=1

# Точка входа
CMD ["python", "run_system.py"]
