"""Часть 3: Многопоточная и многопроцессорная обработка."""

from .augment import ImageAugmenter, augment_worker
from .producer_consumer import Pipeline
from .performance import measure, sequential, parallel

__all__ = ['ImageAugmenter', 'augment_worker', 'Pipeline',
           'measure', 'sequential', 'parallel']