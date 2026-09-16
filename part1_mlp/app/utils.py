"""
Вспомогательные функции: генерация тестовых данных и метрики.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def create_sample_data(n_samples: int = 500,
                       n_features: int = 4,
                       noise: float = 0.1,
                       seed: int = 42) -> tuple:
    """
    Генерация синтетических данных.

    Args:
        n_samples: количество образцов
        n_features: количество признаков
        noise: уровень шума
        seed: зерно генератора

    Returns:
        (X, y)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    w = rng.standard_normal((n_features, 1))
    y = np.tanh(X @ w) + rng.standard_normal((n_samples, 1)) * noise
    return X, y


def save_sample_csv(filepath: str,
                    n_samples: int = 500,
                    n_features: int = 4) -> None:
    """Сохранить синтетические данные в CSV-файл."""
    X, y = create_sample_data(n_samples, n_features)
    cols = [f'f{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Среднеквадратичная ошибка."""
    return float(np.mean((y_true - y_pred) ** 2))


def accuracy(y_true: np.ndarray,
             y_pred: np.ndarray,
             threshold: float = 0.5) -> float:
    """Точность бинарной классификации в процентах."""
    yt = (y_true > threshold).astype(int)
    yp = (y_pred > threshold).astype(int)
    return float(np.mean(yt == yp) * 100)