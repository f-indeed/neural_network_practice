"""Producer-Consumer пайплайн: потоки + процессы."""

import os
import time
import threading
import multiprocessing as mp
from queue import Queue, Empty
from pathlib import Path
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


# ========== Отдельные функции для процессов ==========
def aug_worker_process(data_q, result_q, config):
    """Функция аугментации (отдельная, для процесса)."""
    from app.augment import ImageAugmenter

    pid = os.getpid()
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [PID {pid}] - %(message)s',
        datefmt='%H:%M:%S',
        force=True
    )
    logger_proc = logging.getLogger(__name__)
    logger_proc.info(f"Aug-Worker PID={pid}: старт")

    aug = ImageAugmenter(config)
    count = 0

    while True:
        try:
            data = data_q.get(timeout=5)
            if data is None:
                break
            augmented = aug.augment(data)
            result_q.put((pid, len(augmented), augmented))
            count += 1
        except Empty:
            break

    logger_proc.info(f"Aug-Worker PID={pid}: обработано {count}")
    result_q.put(('DONE', pid, count))


class Pipeline:
    """Пайплайн: Producer → Consumers → Aug-workers."""

    def __init__(self, config):
        self.config = config
        self.file_q = Queue(maxsize=100)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.stats = {'found': 0, 'processed': 0, 'augmented': 0}

    def producer(self, path, extensions, file_q):
        logger.info(f"Producer: сканируем {path}")
        try:
            for ext in extensions:
                ext = ext.lstrip('.')
                for f in Path(path).rglob(f'*.{ext}'):
                    if self.stop.is_set():
                        break
                    file_q.put(str(f))
                    with self.lock:
                        self.stats['found'] += 1
        finally:
            file_q.put(None)
            logger.info(f"Producer завершён: найдено {self.stats['found']}")

    def consumer(self, cid, file_q, data_q):
        logger.info(f"Consumer-{cid}: старт")
        count = 0
        while not self.stop.is_set():
            try:
                fp = file_q.get(timeout=1)
                if fp is None:
                    file_q.put(None)
                    break
                arr = self._read_file(fp)
                if arr is not None:
                    data_q.put(arr)
                    count += 1
                    with self.lock:
                        self.stats['processed'] += 1
            except Empty:
                continue
        logger.info(f"Consumer-{cid}: обработано {count}")

    def _read_file(self, fp):
        try:
            fp = Path(fp)
            suffix = fp.suffix.lower()
            if suffix in ['.jpg', '.jpeg', '.png', '.bmp']:
                img = Image.open(fp).convert('RGB')
                if max(img.size) > 128:
                    img = img.resize((128, 128))
                return np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            logger.error(f"Ошибка {fp}: {e}")
        return None

    def run(self, path, extensions, n_consumers=3, n_processes=2):
        start = time.time()

        # Очереди создаются ЗДЕСЬ, а не в __init__
        file_q = Queue(maxsize=100)
        data_q = mp.Queue(maxsize=100)
        result_q = mp.Queue(maxsize=100)

        logger.info("=" * 50)
        logger.info("СТАРТ ПАЙПЛАЙНА")
        logger.info("=" * 50)

        # Producer
        prod = threading.Thread(
            target=self.producer,
            args=(path, extensions, file_q),
            name='Producer'
        )
        prod.start()

        # Consumers
        consumers = []
        for i in range(n_consumers):
            t = threading.Thread(
                target=self.consumer,
                args=(i, file_q, data_q),
                name=f'Consumer-{i}'
            )
            t.start()
            consumers.append(t)

        # Augmentation processes
        procs = []
        for _ in range(n_processes):
            p = mp.Process(
                target=aug_worker_process,
                args=(data_q, result_q, self.config)
            )
            p.start()
            procs.append(p)

        # Ждём завершения потоков
        prod.join()
        for t in consumers:
            t.join()

        # Стоп-сигналы процессам
        for _ in range(n_processes):
            data_q.put(None)

        # Сбор результатов
        results = []
        done_count = 0
        while done_count < n_processes:
            try:
                msg = result_q.get(timeout=15)
                if msg[0] == 'DONE':
                    done_count += 1
                else:
                    pid, n_aug, augmented = msg
                    results.extend(augmented)
                    self.stats['augmented'] += n_aug
            except Empty:
                break

        # Завершение процессов
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

        elapsed = time.time() - start
        self.stats['time'] = elapsed

        logger.info("=" * 50)
        logger.info("ПАЙПЛАЙН ЗАВЕРШЁН")
        logger.info(f"Найдено: {self.stats['found']}")
        logger.info(f"Обработано: {self.stats['processed']}")
        logger.info(f"Аугментировано: {self.stats['augmented']}")
        logger.info(f"Время: {elapsed:.2f} сек")
        logger.info("=" * 50)

        return {'stats': self.stats, 'results': results}