"""CLI: команды prepare и doctor."""

import argparse
import sys
import os
import subprocess
import platform
import numpy as np
from pathlib import Path
from dotenv import load_dotenv   # ← добавили

from .config import Config
from .scanner import FileScanner
from .logging_config import setup_logging
import logging

# Загружаем .env при импорте модуля
load_dotenv()   # ← добавили

logger = logging.getLogger(__name__)


def prepare_command(args) -> int:
    """
    Команда prepare: сканирование и преобразование файлов.

    Args:
        args: Аргументы командной строки.

    Returns:
        Код возврата (0 — успех).
    """
    print("=" * 60)
    print("📁 ПОДГОТОВКА ДАННЫХ".center(60))
    print("=" * 60)

    config = Config()
    log_level = args.log_level or config.get('log_level', 'INFO')
    setup_logging(log_level, config.get('log_file'))

    try:
        path = Path(args.path)
        if not path.exists():
            logger.error(f"Путь не существует: {path}")
            return 1

        # Сканирование
        scanner = FileScanner(config.data)
        extensions = [e.strip() for e in args.ext.split(',')]
        files = scanner.scan(args.path, extensions)

        if not files:
            print("⚠️ Файлы не найдены")
            return 0

        # Обработка
        arrays = scanner.process_all(files)

        if not arrays:
            print("❌ Не удалось обработать файлы")
            return 1

        # Объединение
        try:
            combined = np.stack(arrays)
        except ValueError:
            logger.warning("Разные размеры — сохраняем как object array")
            combined = np.array(arrays, dtype=object)

        # Сохранение
        output = args.output or config.get('default_output', 'data.npy')
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, combined)

        # Статистика
        print("-" * 60)
        print(f"✅ Обработано: {len(arrays)} файлов")
        print(f"   Ошибок: {len(scanner.errors)}")
        print(f"   Размер: {scanner.stats['total_size']/1024/1024:.2f} MB")
        print(f"   Время: {scanner.stats['time']:.2f} сек")
        print(f"   Сохранено: {output_path}")
        print(f"   Форма: {combined.shape}")
        print("-" * 60)

        return 0

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        return 1


def doctor_command(args) -> int:
    """
    Команда doctor: проверка окружения.

    Args:
        args: Аргументы командной строки.

    Returns:
        Код возврата.
    """
    print("=" * 60)
    print("🩺 ПРОВЕРКА ОКРУЖЕНИЯ".center(60))
    print("=" * 60)

    # Python
    print(f"\n🐍 Python: {sys.version.split()[0]}")
    print(f"   Путь: {sys.executable}")
    print(f"   ОС: {platform.system()} {platform.release()}")

    # Библиотеки
    print("\n📦 Библиотеки:")
    for name in ['numpy', 'PIL', 'pandas', 'sklearn', 'matplotlib']:
        try:
            m = __import__(name)
            version = getattr(m, '__version__', '?')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ❌ {name}: не установлена")

    # CUDA
    print("\n🎮 CUDA:")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"   ✅ {result.stdout.strip()}")
        else:
            print("   ❌ GPU не найдена")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("   ⚠️ nvidia-smi не доступен")

    # Директория
    print(f"\n📂 Рабочая директория: {os.getcwd()}")

    # ENV
    print("\n🌍 Переменные окружения:")
    for var in ['DATA_PATH', 'OUTPUT_PATH', 'LOG_LEVEL']:
        print(f"   {var} = {os.getenv(var, 'не задана')}")

    print("=" * 60)
    print("✅ Проверка завершена!")
    return 0


def main() -> int:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(
        description='Утилита подготовки данных для нейросетей'
    )
    subparsers = parser.add_subparsers(dest='command', help='Команды')

    # prepare
    p = subparsers.add_parser('prepare', help='Подготовка данных')
    p.add_argument('--path', '-p', required=True,
                   help='Путь к директории с данными')
    p.add_argument('--ext', '-e', required=True,
                   help='Расширения файлов через запятую (.jpg,.png,.csv)')
    p.add_argument('--output', '-o', default=None,
                   help='Выходной .npy файл')
    p.add_argument('--log-level', default=None,
                   choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    # doctor
    subparsers.add_parser('doctor', help='Проверка окружения')

    args = parser.parse_args()

    if args.command == 'prepare':
        return prepare_command(args)
    elif args.command == 'doctor':
        return doctor_command(args)

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())