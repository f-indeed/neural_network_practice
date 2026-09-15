"""Аугментация изображений: повороты, отражения, шум."""

import numpy as np
from scipy.ndimage import rotate


class ImageAugmenter:
    """Аугментация изображений."""

    def __init__(self, config=None):
        """Инициализация с настройками."""
        config = config or {}
        self.rotation_range = config.get('rotation_range', 20)
        self.noise_std = config.get('noise_std', 0.05)

    def augment(self, image):
        """
        Применить аугментацию к изображению.

        Returns:
            Список аугментированных изображений.
        """
        result = [image.copy()]

        # Поворот
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range,
                                      self.rotation_range)
            result.append(self._rotate(image, angle))

        # Горизонтальное отражение (50% вероятность)
        if np.random.random() > 0.5:
            result.append(np.fliplr(image).copy())

        # Шум
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, image.shape)
            result.append(np.clip(image + noise, 0, 1))

        return result

    def _rotate(self, image, angle):
        """Поворот изображения на угол (градусы)."""
        if image.ndim == 3:
            out = np.zeros_like(image)
            for c in range(image.shape[2]):
                out[:, :, c] = rotate(image[:, :, c], angle,
                                      reshape=False, order=1)
            return out
        return rotate(image, angle, reshape=False, order=1)


def augment_worker(args):
    """
    Рабочая функция для ProcessPoolExecutor.

    Args:
        args: Кортеж (image, config).

    Returns:
        Список аугментированных изображений.
    """
    img, config = args
    return ImageAugmenter(config).augment(img)