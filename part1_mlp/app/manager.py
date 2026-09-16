"""
Класс DatasetManager — загрузка CSV, нормализация, разделение выборки.
"""

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .exceptions import DataLoadError


class DatasetManager:
    """
    Менеджер датасета: загрузка из CSV, нормализация, разбиение на train/test.
    """

    def __init__(self):
        """Инициализация пустого менеджера."""
        self.features: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.target_name: str = ''
        self.scaler = None

    def load_csv(self, filepath: str,
                 target_column: str = 'last'
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Загрузка данных из CSV.

        Args:
            filepath: путь к файлу
            target_column: 'last' или имя целевой колонки

        Returns:
            (features, target)

        Raises:
            DataLoadError: при ошибке чтения файла
        """
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise DataLoadError(f"Не удалось загрузить CSV: {e}") from e

        if df.empty:
            raise DataLoadError("CSV пустой")

        if target_column == 'last':
            self.target_name = df.columns[-1]
        elif target_column in df.columns:
            self.target_name = target_column
        else:
            raise DataLoadError(
                f"Колонка '{target_column}' не найдена"
            )

        self.feature_names = [
            c for c in df.columns if c != self.target_name
        ]
        self.features = df[self.feature_names].values
        self.target = df[self.target_name].values.reshape(-1, 1)

        print(f"Загружено: {self.features.shape[0]} строк, "
              f"{self.features.shape[1]} признаков")
        return self.features, self.target

    def normalize(self, method: str = 'standard') -> np.ndarray:
        """
        Нормализация признаков.

        Args:
            method: 'standard' (Z-оценка) или 'minmax' (0..1)

        Returns:
            нормализованные признаки
        """
        if self.features is None:
            raise DataLoadError("Данные не загружены")

        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        self.features = self.scaler.fit_transform(self.features)
        return self.features

    def split(self, test_size: float = 0.2,
              random_state: int = 42
              ) -> Tuple[np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray]:
        """
        Разбиение выборки на обучающую и тестовую.

        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.features is None or self.target is None:
            raise DataLoadError("Данные не загружены")

        return train_test_split(
            self.features, self.target,
            test_size=test_size, random_state=random_state
        )

    def info(self) -> dict:
        """Краткая информация о датасете."""
        if self.features is None:
            return {'loaded': False}
        return {
            'loaded': True,
            'n_samples': int(self.features.shape[0]),
            'n_features': int(self.features.shape[1]),
            'feature_names': self.feature_names,
            'target_name': self.target_name,
        }