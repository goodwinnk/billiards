import torch
import cv2

from ball_detection.candidate_classifier.model import Net, CLASSIFICATION_SCORE_THRESHOLD
from ball_detection.candidate_classifier.data_preprocessing import cut_boxes


def visualize_balls(image, balls):
    visualization = image.copy()
    if not len(balls):
        return visualization
    for _, (x0, x1, y0, y1) in balls:
        cv2.rectangle(visualization, (x0, y0), (x1, y1), (0, 255, 0), 3)
    return visualization


class BallDetector:
    def __init__(self, candidate_generator, net_path='ball_detection/candidate_classifier/weights.pt'):
        self.candidate_generator = candidate_generator
        self.classifier = Net()
        self.classifier.load_state_dict(torch.load(net_path))

    def _classify(self, boxes):
        boxes = torch.Tensor(boxes)
        boxes = boxes.permute((0, 3, 1, 2))
        classification_scores = self.classifier(boxes)
        classification_scores = classification_scores.detach().numpy().flatten()
        prediction = classification_scores > CLASSIFICATION_SCORE_THRESHOLD
        return prediction

    def get_balls(self, image):
        candidate_regions = self.candidate_generator.get_regions(image)
        box_cuts = cut_boxes(image, (box for _, box in candidate_regions))
        if not len(box_cuts):
            return []
        classification_mask = self._classify(box_cuts)
        return [region for region, prediction in zip(candidate_regions, classification_mask) if prediction]
