import cv2
import numpy as np

from ball_detection.commons import BallType, BallRegion


VIS_COLORS = {
    BallType.INTEGRAL: (0, 255, 0),
    BallType.STRIPED: (255, 0, 0),
    BallType.WHITE: (255, 255, 255),
    BallType.BLACK: (0, 0, 0)
}


def visualize_balls(image: np.ndarray, balls: BallRegion) -> np.ndarray:
    visualization = image.copy()
    return visualize_balls_on_image(visualization, balls)


def visualize_balls_on_image(image: np.ndarray, balls: BallRegion) -> np.ndarray:
    if not len(balls):
        return image
    for _, box, ball_type in balls:
        cv2.rectangle(image, (box.x0, box.y0), (box.x1, box.y1), VIS_COLORS[ball_type], 3)
    return image
