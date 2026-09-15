"""Замер производительности: последовательно vs параллельно."""

import time
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

from .augment import ImageAugmenter, augment_worker


def sequential(images, config):
    """Последовательная аугментация."""
    aug = ImageAugmenter(config)
    result = []
    for img in images:
        result.extend(aug.augment(img))
    return result


def parallel(images, config):
    """Параллельная аугментация через ProcessPoolExecutor."""
    args = [(img, config) for img in images]
    with ProcessPoolExecutor(max_workers=min(len(images), cpu_count())) as ex:
        results = list(ex.map(augment_worker, args))

    flat = []
    for r in results:
        flat.extend(r)
    return flat


def measure(images, config, repetitions=3):
    """
    Сравнение производительности.

    Args:
        images: Список изображений.
        config: Конфигурация аугментации.
        repetitions: Количество повторов.

    Returns:
        Словарь с результатами.
    """
    print("=" * 50)
    print("ЗАМЕР ПРОИЗВОДИТЕЛЬНОСТИ")
    print(f"Изображений: {len(images)}, повторов: {repetitions}")
    print("=" * 50)

    seq_times, par_times = [], []

    # Последовательная
    print("\n▶ Последовательная обработка:")
    for i in range(repetitions):
        t = time.time()
        sequential(images, config)
        dt = time.time() - t
        seq_times.append(dt)
        print(f"  Попытка {i+1}: {dt:.3f} сек")

    # Параллельная
    print("\n▶ Параллельная обработка:")
    for i in range(repetitions):
        t = time.time()
        parallel(images, config)
        dt = time.time() - t
        par_times.append(dt)
        print(f"  Попытка {i+1}: {dt:.3f} сек")

    # Итоги
    seq_avg = np.mean(seq_times)
    par_avg = np.mean(par_times)
    speedup = seq_avg / par_avg if par_avg > 0 else 0

    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'images_count': len(images),
        'cpu_count': cpu_count(),
        'sequential_avg': round(seq_avg, 4),
        'parallel_avg': round(par_avg, 4),
        'speedup': round(speedup, 2),
        'efficiency_percent': round(speedup / cpu_count() * 100, 1)
    }

    # Сохранение
    with open('performance_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("ИТОГИ:")
    print(f"  Последовательно: {seq_avg:.3f} сек")
    print(f"  Параллельно:     {par_avg:.3f} сек")
    print(f"  Ускорение:       {speedup:.2f}x")
    print(f"  Эффективность:   {report['efficiency_percent']}%")
    print(f"  CPU:             {cpu_count()} ядер")
    print(f"  Отчёт:           performance_report.json")
    print("=" * 50)

    return report