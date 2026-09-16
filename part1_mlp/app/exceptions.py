"""
Пользовательские классы исключений для нейронной сети.
"""


class NeuralNetworkError(Exception):
    """Базовое исключение для ошибок нейронной сети."""


class InvalidLayerSizeError(NeuralNetworkError):
    """Неверные размеры слоёв (например, отрицательные или < 2 слоёв)."""


class MismatchedDataError(NeuralNetworkError):
    """Несоответствие размерностей данных архитектуре сети."""


class ModelNotTrainedError(NeuralNetworkError):
    """Попытка использовать необученную модель."""


class FileLoadError(NeuralNetworkError):
    """Ошибка при работе с файлами (чтение/запись)."""


class ActivationFunctionError(NeuralNetworkError):
    """Неподдерживаемая функция активации."""


class DataLoadError(NeuralNetworkError):
    """Ошибка при загрузке или обработке данных."""