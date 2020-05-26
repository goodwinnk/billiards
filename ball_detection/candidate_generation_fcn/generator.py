from typing import List

import numpy as np
import cv2
import torch

from ball_detection.commons import CandidateGenerator, BallCandidate, Point, Rectangle, fit_region_in_image
from ball_detection.candidate_generation_fcn.model import BallLocationFCN


PREDICTION_THRESHOLD = .5
MIN_BALL_CONTOUR_LEN = 8
MAX_BALL_CONTOUR_LEN = 40
MIN_RAD = 8
RADIUS_SCALE_COEFFICIENT = 1.5
IMAGE_SIZE_RATIO = 2
MASK_SIZE_RATIO = 4
BOX_SCALE_FACTOR = 1.7


class FCNCandidateGenerator(CandidateGenerator):
    def __init__(self, weights_path: str, table_mask: np.ndarray, device='cpu'):
        self.net = BallLocationFCN().to(device)
        self.net.load_state_dict(torch.load(weights_path))
        self.device = device
        self.table_mask = table_mask[::MASK_SIZE_RATIO, ::MASK_SIZE_RATIO]

    def get_regions(self, image: np.ndarray) -> List[BallCandidate]:
        net_image = torch.FloatTensor(image[::IMAGE_SIZE_RATIO, ::IMAGE_SIZE_RATIO]) / 255
        net_image = net_image.permute((2, 0, 1)).unsqueeze(0)
        net_image = net_image.to(self.device)
        net_prediction = self.net(net_image).cpu().detach().numpy().squeeze()
        ball_mask = np.uint8(net_prediction > PREDICTION_THRESHOLD) * self.table_mask
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        image_resolution = image.shape[:2]
        regions = []
        for contour in contours:
            if MIN_BALL_CONTOUR_LEN <= len(contour) <= MAX_BALL_CONTOUR_LEN:
                contour *= MASK_SIZE_RATIO
                cx, cy = contour.mean(axis=0).flatten()
                cx, cy = int(cx), int(cy)
                center = Point(cx, cy)
                xs, ys = contour.squeeze().T
                max_coord_delta = max(np.abs(ys - cy).max(), np.abs(xs - cx).max())
                radius = max(max_coord_delta * RADIUS_SCALE_COEFFICIENT, MIN_RAD)
                box = fit_region_in_image(radius, center, image_resolution)
                regions.append(BallCandidate(center, box))
        return regions
