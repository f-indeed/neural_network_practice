# Часть 1. Многослойный перцептрон (ООП)

Объектно-ориентированная реализация полносвязной нейронной сети
с обучением методом обратного распространения ошибки.

## Структура

    part1_mlp/
    ├── app/
    │   ├── __init__.py
    │   ├── exceptions.py   # пользовательские исключения
    │   ├── models.py       # класс NeuralNetwork
    │   ├── manager.py      # класс DatasetManager
    │   ├── utils.py        # вспомогательные функции
    │   └── main.py         # консольное меню
    ├── README.md
    └── sample_data.csv

## Запуск

```bash
cd part1_mlp
python -m app.main