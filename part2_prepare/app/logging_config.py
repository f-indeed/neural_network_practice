"""Настройка логирования."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = 'INFO',
                  log_file: Optional[str] = None) -> None:
    """
    Настройка логирования в консоль и файл.

    Args:
        log_level: Уровень логирования.
        log_file: Путь к файлу лога.
    """
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }

    logger = logging.getLogger()
    logger.setLevel(levels.get(log_level.upper(), logging.INFO))

    # Очищаем старые обработчики
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fmt = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Консоль
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Файл
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)