from itertools import chain

import torch
import cv2

from ball_detection.candidate_classifier.model import Net
from ball_detection.candidate_classifier.data_preprocessing import cut_boxes
from ball_detection.utils import BallType

VIS_COLORS = {
    BallType.INTEGRAL: (0, 255, 0),
    BallType.STRIPED: (255, 0, 0),
    BallType.WHITE: (255, 255, 255),
    BallType.BLACK: (0, 0, 0)
}


def visualize_balls(image, balls):
    visualization = image.copy()
    if not len(balls):
        return visualization
    for _, (x0, x1, y0, y1), ball_type in balls:
        cv2.rectangle(visualization, (x0, y0), (x1, y1), VIS_COLORS[ball_type], 3)
    return visualization


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
