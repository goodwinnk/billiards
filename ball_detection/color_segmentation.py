import numpy as np


def mask_blue(hue, lightness, saturation):
    return (99 <= hue) & (hue <= 106) & (lightness <= 100)


def mask_yellow(hue, lightness, saturation):
    return (20 <= hue) & (hue <= 40)


def mask_red(hue, lightness, saturation):
    return ((hue <= 3) | (165 <= hue)) & (saturation >= 155)


def mask_orange(hue, lightness, saturation):
    return (4 <= hue) & (hue <= 18) & (lightness <= 220) & (saturation <= 220)


def mask_green(hue, lightness, saturation):
    return (40 <= hue) & (hue <= 87) & (saturation >= 220) & (lightness <= 120)


def mask_crimson(hue, lightness, saturation):
    return (160 <= hue) & (hue <= 173) & (saturation <= 150)


def mask_purple(hue, lightness, saturation):
    return (109 <= hue) & (hue <= 150)


def mask_white(image):
    return np.linalg.norm(image, axis=2) > 360


def mask_black(image):
    return np.linalg.norm(image, axis=2) < 100
