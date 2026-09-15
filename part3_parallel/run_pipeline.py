"""Запуск Части 3: пайплайн и замер производительности."""

import logging
import numpy as np
from pathlib import Path
from PIL import Image

from app.producer_consumer import Pipeline
from app.performance import measure

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_test_images(folder='images', n=20, size=64):
    Path(folder).mkdir(exist_ok=True)
    np.random.seed(42)
    for i in range(n):
        img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(img).save(f'{folder}/img_{i:02d}.png')
    print(f"OK: создано {n} изображений в {folder}/")


if __name__ == '__main__':
    create_test_images('images', n=20, size=64)

    config = {'rotation_range': 20, 'noise_std': 0.05}

    print("\n" + "=" * 50)
    print("ЧАСТЬ 3. ЗАПУСК ПАЙПЛАЙНА")
    print("=" * 50)

    pipe = Pipeline(config)
    result = pipe.run('images', ['.png'],
                      n_consumers=3, n_processes=2)

    print(f"\nОбработано: {result['stats']['processed']}")
    print(f"Аугментировано: {result['stats']['augmented']}")

    print("\n" + "=" * 50)
    print("ЗАМЕР ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 50)

    images = []
    for f in sorted(Path('images').glob('*.png')):
        img = Image.open(f).convert('RGB')
        images.append(np.array(img, dtype=np.float32) / 255.0)

    report = measure(images, config, repetitions=3)