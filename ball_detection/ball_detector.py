from itertools import chain
from collections import deque
from typing import List

import torch
import numpy as np

from ball_detection.candidate_classifier.model import Net
from ball_detection.candidate_classifier.data_preprocessing import cut_boxes
from ball_detection.commons import BallType, BallRegion, iou, CandidateGenerator


# The distance metric is l0
DUPLICATE_CENTER_MAX_DIST = 10
DUPLICATE_MAX_IOU = .5


def get_unique_balls(balls: List[BallRegion]) -> List[BallRegion]:
    balls = sorted(balls, key=lambda b: b.center.y)
    close_x_balls = deque()
    unique_balls = []
    for ball in balls:
        while len(close_x_balls) and close_x_balls[0].center.y < ball.center.y - DUPLICATE_CENTER_MAX_DIST:
            close_x_balls.popleft()
        duplicate = False
        for close_x_ball in close_x_balls:
            if abs(close_x_ball.center.x - ball.center.x) <= DUPLICATE_CENTER_MAX_DIST:
                if iou(close_x_ball.box, ball.box) > DUPLICATE_MAX_IOU:
                    duplicate = True
                    break
        if not duplicate:
            unique_balls.append(ball)
        close_x_balls.append(ball)
    return unique_balls


class BallDetector:
    def __init__(self, candidate_generators: List[CandidateGenerator],
                 net_path: str = 'ball_detection/candidate_classifier/weights.pt'):
        self.candidate_generators = candidate_generators
        self.classifier = Net()
        self.classifier.load_state_dict(torch.load(net_path))

    def _classify(self, boxes: np.ndarray) -> np.ndarray:
        boxes = torch.Tensor(boxes)
        boxes = boxes.permute((0, 3, 1, 2))
        classification_scores = self.classifier(boxes)
        classification_scores = classification_scores.detach().numpy()
        prediction = classification_scores.argmax(axis=1)
        return prediction

    def get_balls(self, image: np.ndarray) -> List[BallRegion]:
        candidate_regions = list(chain(
            *(candidate_generator.get_regions(image) for candidate_generator in self.candidate_generators)
        ))
        box_cuts = cut_boxes(image, (region.box for region in candidate_regions))
        if not len(box_cuts):
            return []
        region_classes = self._classify(box_cuts)
        return [
            BallRegion(region.center, region.box, BallType(prediction))
            for region, prediction in zip(candidate_regions, region_classes)
            if prediction
        ]
