#!/usr/bin/env python
"""
Entry point for text worker
"""
from rq import Worker, Queue
from redis import Redis
from config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    redis_conn = Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        db=Config.REDIS_DB,
        password=Config.REDIS_PASSWORD
    )

    queue = Queue(Config.TEXT_QUEUE, connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    logger.info(f"Запуск воркера текстовых страниц (очередь: {Config.TEXT_QUEUE})")
    worker.work()
