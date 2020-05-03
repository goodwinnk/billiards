import random

import numpy as np
import cv2


NOISE_STD = .02, .02, .02
NOISE_MEAN = 0, 0, 0
BLUR_INTENSITY_OPTIONS = (3, 3), (5, 5), (7, 7)
COLOR_SHIFT_STD = 20
GAMMA_STD = .5


def _saturate(img):
    img[img < 0.] = 0.
    img[img > 1.] = 1.


def noise(img):
    noise_mixin = cv2.randn(np.zeros_like(img), NOISE_MEAN, NOISE_STD)
    img = img + noise_mixin
    _saturate(img)
    return img


def blur(image):
    return cv2.GaussianBlur(image, random.choice(BLUR_INTENSITY_OPTIONS), 0)


def _fit(data, range_start=0., range_end=360.):
    range_size = range_end - range_start
    data[data > range_end] -= range_size
    data[data < range_start] += range_start


def color(image):
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_shift = np.random.normal(0, 20)
    hls_image[:, :, 0] += color_shift
    _fit(hls_image[:, :, 0])
    image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
    return image


def flip(image):
    return cv2.flip(image, flipCode=1)


def gamma_transformation(image):
    gamma = np.exp(np.random.normal(0, GAMMA_STD))
    return image ** gamma


DEFAULT_AUGMENTATIONS = (
    (noise, .15),
    (blur, .15),
    (color, .1),
    (flip, .3),
    (gamma_transformation, .15)
)


class AugmentationApplier:
    def __init__(self, augmentations=DEFAULT_AUGMENTATIONS):
        self.augmentations = augmentations

    def apply(self, image):
        processed_image = image
        for action, probability in self.augmentations:
            if np.random.sample() < probability:
                processed_image = action(image)
        return processed_image

    def apply_batch(self, images):
        return np.array(list(map(self.apply, images)))