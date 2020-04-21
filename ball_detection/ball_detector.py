import torch

from ball_detection.candidate_generation_hough import get_hough_circles, get_circle_region_borders
from ball_detection.candidate_classifier.model import Net, CLASSIFICATION_SCORE_THRESHOLD
from ball_detection.candidate_classifier.data_preprocessing import cut_boxes


class BallDetector:
    def __init__(self, net_path='ball_detection/candidate_classifier/weights.pt'):
        self.classifier = Net()
        self.classifier.load_state_dict(torch.load(net_path))

    def _classify(self, boxes):
        boxes = torch.Tensor(boxes)
        boxes = boxes.permute((0, 3, 1, 2))
        classification_scores = self.classifier(boxes)
        classification_scores = classification_scores.detach().numpy().flatten()
        print(classification_scores)
        prediction = classification_scores > CLASSIFICATION_SCORE_THRESHOLD
        return prediction

    def get_balls(self, image, table_mask):
        circle_candidates = get_hough_circles(image, table_mask)
        candidate_regions = get_circle_region_borders(image, circle_candidates)
        boxes = cut_boxes(image, (box for _, _, box in candidate_regions))
        classification_mask = self._classify(boxes)
        return circle_candidates[classification_mask]
