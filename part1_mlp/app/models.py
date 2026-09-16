"""
Класс NeuralNetwork — полносвязная нейронная сеть с прямым и обратным
распространением ошибки.
"""

import json
from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

from .exceptions import (
    InvalidLayerSizeError,
    MismatchedDataError,
    ModelNotTrainedError,
    ActivationFunctionError,
    FileLoadError,
)


class NeuralNetwork:
    """
    Полносвязная нейронная сеть с настраиваемыми скрытыми слоями.

    Поддерживает:
        * Xavier-инициализацию весов;
        * прямое и обратное распространение;
        * функции активации: сигмоида, ReLU;
        * мини-батчевый градиентный спуск;
        * сохранение и загрузку весов в JSON.

    Атрибуты:
        layer_sizes: список размеров слоёв
        activation: 'sigmoid' или 'relu'
        weights: список матриц весов
        biases: список векторов смещений
        history: словарь с историей ошибок
        is_trained: флаг обученности модели
    """

    def __init__(self, layer_sizes: List[int], activation: str = 'sigmoid'):
        """
        Инициализация сети.

        Args:
            layer_sizes: размеры слоёв, например [4, 8, 1]
            activation: 'sigmoid' или 'relu'

        Raises:
            InvalidLayerSizeError: если слоёв меньше 2 или размер <= 0
            ActivationFunctionError: если активация не поддерживается
        """
        if not layer_sizes or len(layer_sizes) < 2:
            raise InvalidLayerSizeError("Нужно минимум 2 слоя")
        if any(size <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоёв должны быть > 0")

        self.layer_sizes = list(layer_sizes)
        self.activation = activation.lower()

        if self.activation not in ('sigmoid', 'relu'):
            raise ActivationFunctionError(
                f"Функция '{activation}' не поддерживается"
            )

        self._initialize_weights()
        self.history = {'train_loss': [], 'val_loss': []}
        self.is_trained = False

    def _initialize_weights(self) -> None:
        """Xavier-инициализация весов, нулевые смещения."""
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        for i in range(len(self.layer_sizes) - 1):
            scale = np.sqrt(2.0 / self.layer_sizes[i])
            self.weights.append(
                np.random.randn(self.layer_sizes[i],
                                self.layer_sizes[i + 1]) * scale
            )
            self.biases.append(np.zeros((1, self.layer_sizes[i + 1])))

    # ---------------------------------------------------------------- #
    # Функции активации
    # ---------------------------------------------------------------- #
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Применение функции активации."""
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return np.maximum(0, x)  # ReLU

    def _activation_derivative(self, activated: np.ndarray) -> np.ndarray:
        """Производная функции активации по её выходу."""
        if self.activation == 'sigmoid':
            return activated * (1 - activated)
        return (activated > 0).astype(float)

    # ---------------------------------------------------------------- #
    # Прямой проход
    # ---------------------------------------------------------------- #
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Прямое распространение.

        Args:
            X: признаки (n_samples, n_features)

        Returns:
            Выход сети

        Raises:
            MismatchedDataError: если число признаков не совпадает
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Ожидается {self.layer_sizes[0]} признаков, "
                f"получено {X.shape[1]}"
            )

        self.activations = [X]
        current = X

        for i in range(len(self.weights)):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                # Выходной слой — сигмоида
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            else:
                a = self._activation_function(z)
            self.activations.append(a)
            current = a

        return current

    # ---------------------------------------------------------------- #
    # Обратный проход
    # ---------------------------------------------------------------- #
    def backward(self, X: np.ndarray, y: np.ndarray,
                 learning_rate: float) -> float:
        """
        Обратное распространение с обновлением весов.

        Returns:
            MSE на батче
        """
        output = self.forward(X)
        error = output - y
        mse = float(np.mean(error ** 2))

        deltas = [error * (output * (1 - output))]  # выходной слой

        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * \
                    self._activation_derivative(self.activations[i])
            deltas.append(delta)

        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * \
                np.dot(self.activations[i].T, deltas[i])
            self.biases[i] -= learning_rate * \
                np.sum(deltas[i], axis=0, keepdims=True)

        return mse

    # ---------------------------------------------------------------- #
    # Обучение
    # ---------------------------------------------------------------- #
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              learning_rate: float = 0.01,
              batch_size: int = 32,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Обучение сети мини-батчами.

        Returns:
            словарь с историей ошибок
        """
        n = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': []}
        val_loss = None

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_sh, y_sh = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                Xb = X_sh[i:i + batch_size]
                yb = y_sh[i:i + batch_size]
                epoch_loss += self.backward(Xb, yb, learning_rate)
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            history['train_loss'].append(avg_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = float(np.mean((val_pred - y_val) ** 2))
                history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                v = f", val={val_loss:.5f}" if val_loss is not None else ""
                print(f"Эпоха {epoch + 1}/{epochs}, "
                      f"loss={avg_loss:.5f}{v}")

        self.history = history
        self.is_trained = True
        return history

    # ---------------------------------------------------------------- #
    # Предсказание
    # ---------------------------------------------------------------- #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание для новых данных.

        Raises:
            ModelNotTrainedError: если модель не обучена
        """
        if not self.is_trained:
            raise ModelNotTrainedError("Модель не обучена")
        return self.forward(X)

    # ---------------------------------------------------------------- #
    # Сохранение / загрузка весов
    # ---------------------------------------------------------------- #
    def save_weights(self, filepath: str) -> None:
        """
        Сохранение весов и метаданных в JSON.

        Raises:
            FileLoadError: при ошибке записи
        """
        data = {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'history': self.history,
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            raise FileLoadError(f"Не удалось сохранить веса: {e}") from e
        print(f"Веса сохранены в {filepath}")

    def load_weights(self, filepath: str) -> None:
        """
        Загрузка весов из JSON.

        Raises:
            FileLoadError: при ошибке чтения
            MismatchedDataError: если архитектура не совпадает
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise FileLoadError(f"Не удалось загрузить веса: {e}") from e

        if data['layer_sizes'] != self.layer_sizes:
            raise MismatchedDataError(
                f"Архитектура в файле {data['layer_sizes']} "
                f"не совпадает с текущей {self.layer_sizes}"
            )

        self.weights = [np.asarray(w) for w in data['weights']]
        self.biases = [np.asarray(b) for b in data['biases']]
        self.activation = data.get('activation', self.activation)
        self.history = data.get('history', {'train_loss': [], 'val_loss': []})
        self.is_trained = True
        print(f"Веса загружены из {filepath}")

    # ---------------------------------------------------------------- #
    # График
    # ---------------------------------------------------------------- #
    def plot_history(self) -> None:
        """Построение графика ошибки обучения."""
        if not self.history.get('train_loss'):
            print("Нет истории обучения для отображения.")
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='train')
        if self.history.get('val_loss'):
            plt.plot(self.history['val_loss'], label='val')
        plt.xlabel('Эпоха')
        plt.ylabel('MSE')
        plt.title('История обучения')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def __repr__(self) -> str:
        return (f"NeuralNetwork(layers={self.layer_sizes}, "
                f"activation='{self.activation}', "
                f"trained={self.is_trained})")