"""Конфигурация приложения из переменных окружения."""

import os
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv


class Config:
    """Управление конфигурацией из ENV-переменных."""

    def __init__(self, env_file: Optional[str] = None):
        """
        Инициализация конфигурации.

        Args:
            env_file: Путь к .env файлу (опционально).
        """
        if env_file and Path(env_file).exists():
            load_dotenv(env_file)
        else:
            load_dotenv()

        self.data = {
            'data_path': os.getenv('DATA_PATH', './data/input'),
            'output_path': os.getenv('OUTPUT_PATH', './data/output'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': os.getenv('LOG_FILE', './logs/prepare.log'),
            'max_file_size': int(os.getenv('MAX_FILE_SIZE', 104857600)),
            'default_output': os.getenv('DEFAULT_OUTPUT', 'data.npy'),
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение по ключу."""
        return self.data.get(key, default)