"""Сканирование директорий и преобразование файлов в NumPy."""

import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


class FileScanner:
    """Сканер файлов: изображения и CSV → массивы NumPy."""

    def __init__(self, config: Dict):
        """Инициализация сканера."""
        self.config = config
        self.errors: List[Tuple[str, str]] = []
        self.stats = {
            'total_size': 0,
            'time': 0,
            'sizes': []
        }

    def scan(self, path: str, extensions: List[str],
             recursive: bool = True) -> List[Path]:
        """
        Рекурсивный обход директории.

        Args:
            path: Путь к директории.
            extensions: Список расширений.
            recursive: Обходить ли рекурсивно.

        Returns:
            Список найденных файлов.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Директория не найдена: {path}")

        files = []
        for ext in extensions:
            ext = ext.lstrip('.')
            pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
            files.extend(path_obj.glob(pattern))

        logger.info(f"Найдено файлов: {len(files)}")
        return files

    def process_all(self, files: List[Path]) -> List[np.ndarray]:
        """Обработка всех файлов."""
        start = time.time()
        arrays = []

        for i, filepath in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {filepath.name}")
            arr = self._process_file(filepath)
            if arr is not None:
                arrays.append(arr)
                self.stats['total_size'] += filepath.stat().st_size
                self.stats['sizes'].append(arr.shape)

        self.stats['time'] = time.time() - start
        return arrays

    def _process_file(self, filepath: Path) -> Optional[np.ndarray]:
        """Обработка одного файла."""
        try:
            suffix = filepath.suffix.lower()

            if suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
                return self._process_image(filepath)
            elif suffix == '.csv':
                return self._process_csv(filepath)
            else:
                logger.warning(f"Пропущен: {suffix}")
                return None

        except Exception as e:
            logger.error(f"Ошибка {filepath}: {e}")
            self.errors.append((str(filepath), str(e)))
            return None

    def _process_image(self, filepath: Path) -> np.ndarray:
        """Изображение → нормализованный массив [0,1]."""
        img = Image.open(filepath).convert('RGB')

        max_size = 256
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def _process_csv(self, filepath: Path) -> np.ndarray:
        """CSV → массив."""
        return np.loadtxt(filepath, delimiter=',', dtype=np.float32)