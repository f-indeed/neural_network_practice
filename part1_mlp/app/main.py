"""
Консольное меню для работы с нейронной сетью.

Запуск: python -m app.main
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .models import NeuralNetwork
from .manager import DatasetManager
from .exceptions import NeuralNetworkError
from .utils import save_sample_csv


def menu() -> None:
    """Консольное меню для работы с нейронной сетью."""
    net = None
    dm = None
    X_tr = X_te = y_tr = y_te = None

    print("=" * 60)
    print("НЕЙРОННАЯ СЕТЬ — ЧАСТЬ 1")
    print("=" * 60)

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
            # ============ 1. СОЗДАТЬ СЕТЬ ============
            if c == '1':
                print("\n--- Создание сети ---")
                sizes_str = input(
                    "Размеры слоёв через пробел (например, '4 8 4 1'): "
                ).strip()
                if not sizes_str:
                    print("Размеры не введены")
                    continue
                try:
                    sizes = [int(x) for x in sizes_str.split()]
                except ValueError:
                    print("Размеры должны быть целыми числами")
                    continue
                if len(sizes) < 2:
                    print("Минимум 2 слоя (входной и выходной)")
                    continue

                act = input(
                    "Активация (sigmoid/relu, по умолчанию relu): "
                ).strip()
                act = act if act in ('sigmoid', 'relu') else 'relu'

                net = NeuralNetwork(sizes, activation=act)
                print(f"Сеть создана: {sizes}, активация: {act}")

            # ============ 2. ЗАГРУЗИТЬ ДАННЫЕ ============
            elif c == '2':
                print("\n--- Загрузка данных из CSV ---")
                path = input("Введите путь к CSV-файлу: ").strip()
                path = path.strip('"').strip("'")

                if not path:
                    print("Путь не введён")
                    continue

                if not Path(path).exists():
                    print(f"Файл не найден: {path}")
                    create = input(
                        "Создать тестовые данные? (y/n): "
                    ).strip().lower()
                    if create == 'y':
                        path = 'sample_data.csv'
                        save_sample_csv(path)
                        print(f"Создан файл: {Path(path).absolute()}")
                    else:
                        continue

                target_col = input(
                    "Имя целевой колонки (Enter для последней): "
                ).strip() or 'last'

                dm = DatasetManager()
                dm.load_csv(path, target_column=target_col)
                print(f"Данные загружены")
                print(f"   Форма признаков: {dm.features.shape}")
                print(f"   Форма целевой: {dm.target.shape}")
                print(f"   Признаки: {dm.feature_names}")
                print(f"   Целевая: {dm.target_name}")

                normalize = input(
                    "Нормализовать данные? (y/n, по умолчанию y): "
                ).strip().lower()
                if normalize != 'n':
                    method = input(
                        "Метод (standard/minmax, по умолчанию standard): "
                    ).strip() or 'standard'
                    dm.normalize(method)
                    print(f"Нормализация: {method}")

                ts = input(
                    "Доля тестовой выборки (по умолчанию 0.2): "
                ).strip()
                test_size = float(ts) if ts else 0.2

                X_tr, X_te, y_tr, y_te = dm.split(test_size=test_size)
                print(f"Разделение: train={X_tr.shape}, test={X_te.shape}")

            # ============ 3. ОБУЧИТЬ СЕТЬ ============
            elif c == '3':
                if net is None:
                    print("Сначала создайте сеть (пункт 1)")
                    continue
                if X_tr is None:
                    print("Сначала загрузите данные (пункт 2)")
                    continue

                print("\n--- Обучение сети ---")

                if net.layer_sizes[0] != X_tr.shape[1]:
                    print(f"Входной слой {net.layer_sizes[0]}, "
                          f"а признаков {X_tr.shape[1]}")
                    fix = input(
                        "Исправить автоматически? (y/n): "
                    ).strip().lower()
                    if fix == 'y':
                        net = NeuralNetwork(
                            [X_tr.shape[1]] + net.layer_sizes[1:],
                            activation=net.activation
                        )
                        print(f"Сеть пересоздана: {net.layer_sizes}")
                    else:
                        continue

                ep = int(input("Эпох (по умолчанию 100): ").strip() or "100")
                lr = float(input(
                    "Learning rate (по умолчанию 0.05): "
                ).strip() or "0.05")
                bs = int(input(
                    "Batch size (по умолчанию 16): "
                ).strip() or "16")

                print(f"\nОбучение: {ep} эпох, lr={lr}, batch={bs}")
                print("-" * 60)

                net.train(X_tr, y_tr, X_te, y_te,
                          epochs=ep, learning_rate=lr, batch_size=bs)

                print("-" * 60)
                print("Обучение завершено")
                y_pred = net.predict(X_te)
                print(f"MSE на тесте: {np.mean((y_pred - y_te) ** 2):.6f}")

            # ============ 4. ПРЕДСКАЗАНИЕ ============
            elif c == '4':
                if net is None or not net.is_trained:
                    print("Сначала создайте и обучите сеть")
                    continue

                print("\n--- Предсказание ---")
                print("1. На тестовой выборке")
                print("2. Ввести вручную")
                pred_mode = input("Выбор (1/2): ").strip() or '1'

                if pred_mode == '1':
                    if X_te is None:
                        print("Нет тестовой выборки")
                        continue
                    y_pred = net.predict(X_te)
                    mse_val = np.mean((y_pred - y_te) ** 2)
                    print(f"\nMSE на тесте: {mse_val:.6f}")
                    print(f"\nПервые 10 предсказаний:")
                    print(f"{'№':<4}{'Истина':<12}"
                          f"{'Предсказание':<14}{'Ошибка':<10}")
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
                        print("Введите числа")
                        continue
                    if len(vals) != n_features:
                        print(f"Нужно {n_features} значений, "
                              f"введено {len(vals)}")
                        continue

                    if dm is not None and dm.scaler is not None:
                        vals_arr = dm.scaler.transform([vals])
                    else:
                        vals_arr = [vals]

                    pred = net.predict(np.array(vals_arr))
                    print(f"Предсказание: {pred[0][0]:.6f}")

            # ============ 5. СОХРАНИТЬ ВЕСА ============
            elif c == '5':
                if net is None:
                    print("Сначала создайте сеть")
                    continue
                fp = input(
                    "Файл (по умолчанию weights.json): "
                ).strip() or "weights.json"
                if '.' not in fp:
                    fp += '.json'
                net.save_weights(fp)
                print(f"Веса сохранены: {Path(fp).absolute()}")

            # ============ 6. ЗАГРУЗИТЬ ВЕСА ============
            elif c == '6':
                if net is None:
                    print("Сначала создайте сеть с той же архитектурой")
                    continue
                fp = input(
                    "Файл (по умолчанию weights.json): "
                ).strip() or "weights.json"
                if not Path(fp).exists():
                    print(f"Файл не найден: {fp}")
                    continue
                net.load_weights(fp)
                print(f"Веса загружены: {fp}")

            # ============ 7. ГРАФИК ============
            elif c == '7':
                if net is None or not net.is_trained:
                    print("Сначала создайте и обучите сеть")
                    continue
                net.plot_history()

            # ============ 8. ИНФОРМАЦИЯ ============
            elif c == '8':
                print("\n" + "=" * 60)
                print("ИНФОРМАЦИЯ")
                print("=" * 60)

                if net is None:
                    print("Сеть: не создана")
                else:
                    n_params = sum(w.size + b.size
                                   for w, b in zip(net.weights, net.biases))
                    print(f"Сеть:")
                    print(f"   Архитектура: {net.layer_sizes}")
                    print(f"   Активация: {net.activation}")
                    print(f"   Параметров: {n_params}")
                    print(f"   Обучена: "
                          f"{'да' if net.is_trained else 'нет'}")
                    if net.history.get('train_loss'):
                        print(f"   Последний train loss: "
                              f"{net.history['train_loss'][-1]:.6f}")
                    if net.history.get('val_loss'):
                        print(f"   Последний val loss:   "
                              f"{net.history['val_loss'][-1]:.6f}")

                if dm is None:
                    print("\nДанные: не загружены")
                else:
                    print(f"\nДанные:")
                    print(f"   Признаков: {dm.features.shape[1]}")
                    print(f"   Образцов: {dm.features.shape[0]}")
                    print(f"   Признаки: {dm.feature_names}")
                    print(f"   Целевая: {dm.target_name}")
                    if X_tr is not None:
                        print(f"   Train: {X_tr.shape}")
                        print(f"   Test:  {X_te.shape}")

                input("\nНажмите Enter для продолжения...")

            # ============ 0. ВЫХОД ============
            elif c == '0':
                print("Выход")
                break

            else:
                print("Неверный выбор. Введите число от 0 до 8.")

        except NeuralNetworkError as e:
            print(f"Ошибка сети: {e}")
        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except KeyboardInterrupt:
            print("\nПрервано")
            break
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")


def main() -> int:
    """Точка входа."""
    try:
        menu()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    return 0


if __name__ == '__main__':
    sys.exit(main())