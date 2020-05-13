from typing import List

import numpy as np
import cv2
import torch

from ball_detection.commons import CandidateGenerator, BallCandidate, Point, Rectangle
from ball_detection.candidate_generation_fcn.model import BallLocationFCN


PREDICTION_THRESHOLD = .5
MIN_BALL_CONTOUR_LEN = 8
MAX_BALL_CONTOUR_LEN = 40
MIN_HALF_SIDE = 12
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
        m, n = image.shape[:2]
        regions = []
        for contour in contours:
            if MIN_BALL_CONTOUR_LEN <= len(contour) <= MAX_BALL_CONTOUR_LEN:
                contour *= MASK_SIZE_RATIO
                c = Point(*map(int, contour.mean(axis=0)[0]))
                xs, ys = contour.squeeze().T
                max_coord_delta = max(np.abs(ys - c.y).max(), np.abs(xs - c.x).max())
                half_side = max(int(max_coord_delta * 2), MIN_HALF_SIDE)
                half_side = min(half_side, n - c.x, c.x, m - c.y, c.y)
                box = Rectangle(c.x - half_side, c.y - half_side, c.x + half_side + 1, c.y + half_side + 1)
                regions.append(BallCandidate(c, box))
        return regions
