from typing import List

import numpy as np
import cv2

from ball_detection.commons import CANDIDATE_PADDING_COEFFICIENT, BallCandidate, BallRegion, CandidateGenerator, \
    Point, Rectangle


class HoughCircleDetector(CandidateGenerator):
    def __init__(self, table_mask: np.ndarray):
        self.table_mask = table_mask

    def get_hough_circles(self, image: np.ndarray) -> np.ndarray:
        hough_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hough_img = hough_img * self.table_mask
        circles = cv2.HoughCircles(hough_img, cv2.HOUGH_GRADIENT, minDist=15, dp=1.6, param1=100, param2=27,
                                   minRadius=8, maxRadius=25)
        return circles[0] if circles is not None else np.array([], dtype=np.float32).reshape((0, 3))

    def get_regions(self, image: np.ndarray) -> List[BallRegion]:
        regions = []
        m, n = image.shape[:2]
        for x, y, r in self.get_hough_circles(image):
            half_side = min(r * CANDIDATE_PADDING_COEFFICIENT, n - x, x, m - y, y)
            center = Point(x, y)
            box = Rectangle(int(x - half_side), int(y - half_side), int(x + half_side + 1), int(y + half_side + 1))
            regions.append(BallCandidate(center, box, r))
        return regions
