from itertools import chain
from collections import deque

import torch
import cv2

from ball_detection.candidate_classifier.model import Net
from ball_detection.candidate_classifier.data_preprocessing import cut_boxes
from ball_detection.commons import BallType

VIS_COLORS = {
    BallType.INTEGRAL: (0, 255, 0),
    BallType.STRIPED: (255, 0, 0),
    BallType.WHITE: (255, 255, 255),
    BallType.BLACK: (0, 0, 0)
}

DUPLICATE_CENTER_MAX_DIST = 10
DUPLICATE_MAX_IOU = .65


def iou(ball1, ball2):
    x10, x11, y10, y11 = ball1[1]
    x20, x21, y20, y21 = ball2[1]
    xi0 = max(x10, x20)
    yi0 = max(y10, y20)
    xi1 = min(x11, x21)
    yi1 = min(y11, y21)
    if xi0 > xi1 or yi0 > yi1:
        return 0.
    a1 = (x11 - x10) * (y11 - y10)
    a2 = (x21 - x20) * (y21 - y20)
    ai = (xi1 - xi0) * (yi1 - yi0)
    au = a1 + a2 - ai
    return ai / au


def get_unique_balls(balls):
    balls = sorted(balls, key=lambda b: b[0][1])
    close_x_balls = deque()
    unique_balls = []
    for ball in balls:
        cx, cy = ball[0]
        while len(close_x_balls) and close_x_balls[0][0][1] < cy - DUPLICATE_CENTER_MAX_DIST:
            close_x_balls.popleft()
        duplicate = False
        for close_x_ball in close_x_balls:
            if abs(ball[0][0] - cx) <= DUPLICATE_CENTER_MAX_DIST:
                if iou(close_x_ball, ball) > DUPLICATE_MAX_IOU:
                    duplicate = True
        if not duplicate:
            unique_balls.append(ball)
        close_x_balls.append(ball)
    return unique_balls


def visualize_balls(image, balls):
    visualization = image.copy()
    return visualize_balls_on_image(visualization, balls)


def visualize_balls_on_image(image, balls):
    if not len(balls):
        return image
    for _, (x0, x1, y0, y1), ball_type in balls:
        cv2.rectangle(image, (x0, y0), (x1, y1), VIS_COLORS[ball_type], 3)
    return image


class BallDetector:
    def __init__(self, candidate_generators, net_path='ball_detection/candidate_classifier/weights.pt'):
        self.candidate_generators = candidate_generators
        self.classifier = Net()
        self.classifier.load_state_dict(torch.load(net_path))

    def _classify(self, boxes):
        boxes = torch.Tensor(boxes)
        boxes = boxes.permute((0, 3, 1, 2))
        classification_scores = self.classifier(boxes)
        classification_scores = classification_scores.detach().numpy()
        prediction = classification_scores.argmax(axis=1)
        return prediction

    def get_balls(self, image):
        candidate_regions = list(chain(
            *(candidate_generator.get_regions(image) for candidate_generator in self.candidate_generators)
        ))
        box_cuts = cut_boxes(image, (box for _, box in candidate_regions))
        if not len(box_cuts):
            return []
        region_classes = self._classify(box_cuts)
        return [
            region + (BallType(prediction),)
            for region, prediction in zip(candidate_regions, region_classes)
            if prediction
        ]
