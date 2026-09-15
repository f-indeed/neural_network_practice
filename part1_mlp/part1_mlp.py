# ============================================================
# ЧАСТЬ 1. Многослойный перцептрон (ООП)
# ============================================================

# ---------- Ячейка 1: Импорты ----------
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------- Ячейка 2: Исключения ----------
class NeuralNetworkError(Exception):
    """Базовое исключение для ошибок нейронной сети."""
    pass

class InvalidLayerSizeError(NeuralNetworkError):
    """Неверные размеры слоёв."""
    pass

class MismatchedDataError(NeuralNetworkError):
    """Несоответствие размерностей данных."""
    pass

class ModelNotTrainedError(NeuralNetworkError):
    """Модель не обучена."""
    pass

class FileLoadError(NeuralNetworkError):
    """Ошибка работы с файлами."""
    pass

class ActivationFunctionError(NeuralNetworkError):
    """Неподдерживаемая функция активации."""
    pass

class DataLoadError(NeuralNetworkError):
    """Ошибка загрузки данных."""
    pass


# ---------- Ячейка 3: Класс NeuralNetwork ----------
class NeuralNetwork:
    """
    Полносвязная нейронная сеть с настраиваемыми скрытыми слоями.
    Поддерживает прямое и обратное распространение, сигмоиду и ReLU.
    """

    def __init__(self, layer_sizes: List[int], activation: str = 'sigmoid'):
        if not layer_sizes or len(layer_sizes) < 2:
            raise InvalidLayerSizeError("Нужно минимум 2 слоя")
        if any(size <= 0 for size in layer_sizes):
            raise InvalidLayerSizeError("Размеры слоёв должны быть > 0")

        self.layer_sizes = layer_sizes
        self.activation = activation.lower()

        if self.activation not in ['sigmoid', 'relu']:
            raise ActivationFunctionError(
                f"Функция '{activation}' не поддерживается"
            )

        # Xavier-инициализация
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            )
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

        self.history = {'train_loss': [], 'val_loss': []}
        self.is_trained = False

    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return np.maximum(0, x)  # ReLU

    def _activation_derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        return (x > 0).astype(float)  # ReLU

    def forward(self, X):
        """Прямой проход."""
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

    def backward(self, X, y, learning_rate):
        """Обратный проход + обновление весов."""
        output = self.forward(X)
        error = output - y
        mse = np.mean(error ** 2)

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

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, learning_rate=0.01, batch_size=32, verbose=True):
        """Обучение сети."""
        n = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_sh, y_sh = X_train[idx], y_train[idx]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, n, batch_size):
                Xb = X_sh[i:i+batch_size]
                yb = y_sh[i:i+batch_size]
                epoch_loss += self.backward(Xb, yb, learning_rate)
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_loss)

            if X_val is not None:
                val_pred = self.forward(X_val)
                val_loss = np.mean((val_pred - y_val) ** 2)
                history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                v = f", val={val_loss:.5f}" if X_val is not None else ""
                print(f"Эпоха {epoch+1}/{epochs}, loss={avg_loss:.5f}{v}")

        self.history = history
        self.is_trained = True
        return history

    def predict(self, X):
        if not self.is_trained:
            raise ModelNotTrainedError("Модель не обучена")
        return self.forward(X)

    def save_weights(self, filepath):
        data = {
            'layer_sizes': self.layer_sizes,
            'activation': self.activation,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'history': self.history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Веса сохранены в {filepath}")

    def load_weights(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        if data['layer_sizes'] != self.layer_sizes:
            raise MismatchedDataError("Размеры слоёв не совпадают")
        self.weights = [np.array(w) for w in data['weights']]
        self.biases = [np.array(b) for b in data['biases']]
        self.history = data.get('history', {})
        self.is_trained = True
        print(f"Веса загружены из {filepath}")

    def plot_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='train')
        if self.history.get('val_loss'):
            plt.plot(self.history['val_loss'], label='val')
        plt.xlabel('Эпоха')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


# ---------- Ячейка 4: Класс DatasetManager ----------
class DatasetManager:
    """Загрузка, нормализация и разделение данных."""

    def __init__(self):
        self.features = None
        self.target = None
        self.feature_names = []
        self.target_name = ''
        self.scaler = None

    def load_csv(self, filepath, target_column='last'):
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise DataLoadError(f"Не удалось загрузить CSV: {e}")

        if target_column == 'last':
            self.target_name = df.columns[-1]
        else:
            self.target_name = target_column

        self.feature_names = [c for c in df.columns if c != self.target_name]
        self.features = df[self.feature_names].values
        self.target = df[self.target_name].values.reshape(-1, 1)

        print(f"Загружено: {self.features.shape[0]} строк, "
              f"{self.features.shape[1]} признаков")
        return self.features, self.target

    def normalize(self, method='standard'):
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        self.features = self.scaler.fit_transform(self.features)
        return self.features

    def split(self, test_size=0.2, random_state=42):
        return train_test_split(
            self.features, self.target,
            test_size=test_size, random_state=random_state
        )


# ---------- Ячейка 5: Создание тестовых данных ----------
np.random.seed(42)
n = 500
X = np.random.randn(n, 4)
y = (np.tanh(X @ np.random.randn(4, 1)) +
     np.random.randn(n, 1) * 0.1)

df = pd.DataFrame(X, columns=[f'f{i}' for i in range(4)])
df['target'] = y
df.to_csv('sample_data.csv', index=False)
print("Тестовые данные сохранены в sample_data.csv")


# ---------- Ячейка 6: Обучение сети ----------
# Загрузка данных
dm = DatasetManager()
dm.load_csv('sample_data.csv', target_column='target')
dm.normalize('standard')
X_train, X_test, y_train, y_test = dm.split(test_size=0.2)

# Создание сети
net = NeuralNetwork([4, 8, 4, 1], activation='relu')
print(f"Сеть: {net.layer_sizes}, активация: {net.activation}")

# Обучение
history = net.train(
    X_train, y_train,
    X_val=X_test, y_val=y_test,
    epochs=200, learning_rate=0.05, batch_size=16
)


# ---------- Ячейка 7: Предсказание ----------
y_pred = net.predict(X_test)
test_mse = np.mean((y_pred - y_test) ** 2)
print(f"Тестовая MSE: {test_mse:.6f}")
print(f"Пример предсказания: {y_pred[0][0]:.4f} "
      f"(истина: {y_test[0][0]:.4f})")


# ---------- Ячейка 8: График ошибки ----------
net.plot_history()


# ---------- Ячейка 9: Сохранение/загрузка весов ----------
net.save_weights('weights.json')

net2 = NeuralNetwork([4, 8, 4, 1], activation='relu')
net2.load_weights('weights.json')
print("Проверка загрузки:", net2.layer_sizes)
# ---------- Ячейка 10: Консольное меню (обновлённое) ----------
def menu():
    """Консольное меню, соответствующее заданию."""
    net = None
    dm = None
    X_tr = X_te = y_tr = y_te = None
    
    print("=" * 60)
    print("НЕЙРОННАЯ СЕТЬ — ЧАСТЬ 1")
    print("=" * 60)
    
    # ---- Основное меню ----
    while True:
        print("\n" + "=" * 60)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 60)
        print("1. Создать сеть")
        print("2. Загрузить данные из CSV")
        print("3. Обучить сеть")
        print("4. Сделать предсказание")
        print("5. Сохранить веса")
        print("6. Загрузить веса")
        print("7. Показать график ошибки")
        print("8. Показать информацию")
        print("0. Выход")
        print("-" * 60)
        
        c = input("Ваш выбор: ").strip()
        
        try:
            # ==================== 1. СОЗДАТЬ СЕТЬ ====================
            if c == '1':
                print("\n--- Создание сети ---")
                sizes_str = input(
                    "Размеры слоёв через пробел (например, '4 8 4 1'): "
                ).strip()
                
                if not sizes_str:
                    print("❌ Размеры не введены")
                    continue
                
                try:
                    sizes = [int(x) for x in sizes_str.split()]
                except ValueError:
                    print("❌ Размеры должны быть целыми числами")
                    continue
                
                if len(sizes) < 2:
                    print("❌ Минимум 2 слоя (входной и выходной)")
                    continue
                
                act = input("Активация (sigmoid/relu, по умолчанию relu): ").strip()
                act = act if act in ('sigmoid', 'relu') else 'relu'
                
                net = NeuralNetwork(sizes, activation=act)
                print(f"✅ Сеть создана: {sizes}, активация: {act}")
            
            # ==================== 2. ЗАГРУЗИТЬ ДАННЫЕ ====================
            elif c == '2':
                print("\n--- Загрузка данных из CSV ---")
                path = input("Введите путь к CSV-файлу: ").strip()
                path = path.strip('"').strip("'")  # убираем кавычки
                
                if not path:
                    print("❌ Путь не введён")
                    continue
                
                # Проверка существования файла
                if not Path(path).exists():
                    print(f"❌ Файл не найден: {path}")
                    create = input("Создать тестовые данные? (y/n): ").strip().lower()
                    if create == 'y':
                        # Создаём тестовые данные
                        np.random.seed(42)
                        n = 500
                        X_temp = np.random.randn(n, 4)
                        y_temp = np.tanh(X_temp @ np.random.randn(4, 1)) + \
                                 np.random.randn(n, 1) * 0.1
                        
                        df = pd.DataFrame(X_temp,
                                          columns=[f'f{i}' for i in range(4)])
                        df['target'] = y_temp
                        
                        path = 'sample_data.csv'
                        df.to_csv(path, index=False)
                        print(f"✅ Создан файл: {Path(path).absolute()}")
                    else:
                        continue
                
                # Загрузка
                target_col = input(
                    "Имя целевой колонки (Enter для последней): "
                ).strip() or 'last'
                
                dm = DatasetManager()
                dm.load_csv(path, target_column=target_col)
                print(f"✅ Данные загружены")
                print(f"   Форма признаков: {dm.features.shape}")
                print(f"   Форма целевой: {dm.target.shape}")
                print(f"   Признаки: {dm.feature_names}")
                print(f"   Целевая: {dm.target_name}")
                
                # Нормализация
                normalize = input(
                    "Нормализовать данные? (y/n, по умолчанию y): "
                ).strip().lower()
                if normalize != 'n':
                    method = input(
                        "Метод (standard/minmax, по умолчанию standard): "
                    ).strip() or 'standard'
                    dm.normalize(method)
                    print(f"✅ Нормализация: {method}")
                
                # Разделение
                ts = input(
                    "Доля тестовой выборки (по умолчанию 0.2): "
                ).strip()
                test_size = float(ts) if ts else 0.2
                
                X_tr, X_te, y_tr, y_te = dm.split(test_size=test_size)
                print(f"✅ Разделение: train={X_tr.shape}, test={X_te.shape}")
            
            # ==================== 3. ОБУЧИТЬ СЕТЬ ====================
            elif c == '3':
                if net is None:
                    print("❌ Сначала создайте сеть (пункт 1)")
                    continue
                if X_tr is None:
                    print("❌ Сначала загрузите данные (пункт 2)")
                    continue
                
                print("\n--- Обучение сети ---")
                
                # Проверка совместимости входного слоя с данными
                if net.layer_sizes[0] != X_tr.shape[1]:
                    print(f"⚠️ Входной слой {net.layer_sizes[0]}, "
                          f"а признаков {X_tr.shape[1]}")
                    fix = input("Исправить автоматически? (y/n): ").strip().lower()
                    if fix == 'y':
                        net = NeuralNetwork(
                            [X_tr.shape[1]] + net.layer_sizes[1:],
                            activation=net.activation
                        )
                        print(f"✅ Сеть пересоздана: {net.layer_sizes}")
                    else:
                        continue
                
                ep = int(input("Эпох (по умолчанию 100): ").strip() or "100")
                lr = float(input("Learning rate (по умолчанию 0.05): ").strip() or "0.05")
                bs = int(input("Batch size (по умолчанию 16): ").strip() or "16")
                
                print(f"\n📊 Обучение: {ep} эпох, lr={lr}, batch={bs}")
                print("-" * 60)
                
                net.train(X_tr, y_tr, X_te, y_te,
                          epochs=ep, learning_rate=lr, batch_size=bs)
                
                print("-" * 60)
                print("✅ Обучение завершено")
                
                # Оценка
                y_pred = net.predict(X_te)
                mse = np.mean((y_pred - y_te) ** 2)
                print(f"📈 MSE на тесте: {mse:.6f}")
            
            # ==================== 4. ПРЕДСКАЗАНИЕ ====================
            elif c == '4':
                if net is None or not net.is_trained:
                    print("❌ Сначала создайте и обучите сеть")
                    continue
                
                print("\n--- Предсказание ---")
                print("1. На тестовой выборке")
                print("2. Ввести вручную")
                
                pred_mode = input("Выбор (1/2): ").strip() or '1'
                
                if pred_mode == '1':
                    if X_te is None:
                        print("❌ Нет тестовой выборки")
                        continue
                    y_pred = net.predict(X_te)
                    mse = np.mean((y_pred - y_te) ** 2)
                    print(f"\n✅ MSE на тесте: {mse:.6f}")
                    print(f"\nПервые 10 предсказаний:")
                    print(f"{'№':<4}{'Истина':<12}{'Предсказание':<14}{'Ошибка':<10}")
                    print("-" * 45)
                    for i in range(min(10, len(y_pred))):
                        err = abs(y_te[i][0] - y_pred[i][0])
                        print(f"{i+1:<4}{y_te[i][0]:<12.4f}"
                              f"{y_pred[i][0]:<14.4f}{err:<10.4f}")
                
                elif pred_mode == '2':
                    n_features = net.layer_sizes[0]
                    print(f"\nВведите {n_features} значений через пробел:")
                    vals_str = input("> ").strip()
                    
                    try:
                        vals = [float(x) for x in vals_str.split()]
                    except ValueError:
                        print("❌ Введите числа")
                        continue
                    
                    if len(vals) != n_features:
                        print(f"❌ Нужно {n_features} значений, "
                              f"введено {len(vals)}")
                        continue
                    
                    # Нормализация, если была
                    if dm is not None and dm.scaler is not None:
                        vals_arr = dm.scaler.transform([vals])
                    else:
                        vals_arr = [vals]
                    
                    pred = net.predict(np.array(vals_arr))
                    print(f"✅ Предсказание: {pred[0][0]:.6f}")
            
            # ==================== 5. СОХРАНИТЬ ВЕСА ====================
            elif c == '5':
                if net is None:
                    print("❌ Сначала создайте сеть")
                    continue
                
                fp = input("Файл (по умолчанию weights.json): ").strip() \
                     or "weights.json"
                
                # Добавляем расширение, если не указано
                if '.' not in fp:
                    fp += '.json'
                
                net.save_weights(fp)
                print(f"✅ Веса сохранены: {Path(fp).absolute()}")
            
            # ==================== 6. ЗАГРУЗИТЬ ВЕСА ====================
            elif c == '6':
                if net is None:
                    print("❌ Сначала создайте сеть с той же архитектурой")
                    continue
                
                fp = input("Файл (по умолчанию weights.json): ").strip() \
                     or "weights.json"
                
                if not Path(fp).exists():
                    print(f"❌ Файл не найден: {fp}")
                    continue
                
                net.load_weights(fp)
                print(f"✅ Веса загружены: {fp}")
            
            # ==================== 7. ГРАФИК ====================
            elif c == '7':
                if net is None or not net.is_trained:
                    print("❌ Сначала создайте и обучите сеть")
                    continue
                net.plot_history()
            
            # ==================== 8. ИНФОРМАЦИЯ ====================
            elif c == '8':
                print("\n" + "=" * 60)
                print("ИНФОРМАЦИЯ")
                print("=" * 60)
                
                if net is None:
                    print("🧠 Сеть: не создана")
                else:
                    n_params = sum(w.size + b.size
                                   for w, b in zip(net.weights, net.biases))
                    print(f"🧠 Сеть:")
                    print(f"   Архитектура: {net.layer_sizes}")
                    print(f"   Активация: {net.activation}")
                    print(f"   Параметров: {n_params}")
                    print(f"   Обучена: {'да' if net.is_trained else 'нет'}")
                    if net.history.get('train_loss'):
                        print(f"   Последний train loss: "
                              f"{net.history['train_loss'][-1]:.6f}")
                    if net.history.get('val_loss'):
                        print(f"   Последний val loss:   "
                              f"{net.history['val_loss'][-1]:.6f}")
                
                if dm is None:
                    print("\n📊 Данные: не загружены")
                else:
                    print(f"\n📊 Данные:")
                    print(f"   Признаков: {dm.features.shape[1]}")
                    print(f"   Образцов: {dm.features.shape[0]}")
                    print(f"   Признаки: {dm.feature_names}")
                    print(f"   Целевая: {dm.target_name}")
                    if X_tr is not None:
                        print(f"   Train: {X_tr.shape}")
                        print(f"   Test:  {X_te.shape}")
                
                input("\nНажмите Enter для продолжения...")
            
            # ==================== 0. ВЫХОД ====================
            elif c == '0':
                print("👋 Выход")
                break
            
            else:
                print("❌ Неверный выбор. Введите число от 0 до 8.")
        
        except NeuralNetworkError as e:
            print(f"❌ Ошибка сети: {e}")
        except ValueError as e:
            print(f"❌ Ошибка ввода: {e}")
        except KeyboardInterrupt:
            print("\n👋 Прервано")
            break
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")


if __name__ == '__main__':
    menu()