import numpy as np
import cv2


def visualize_circles(image, circles):
    visualization = image.copy()
    if not len(circles):
        return visualization
    for x, y, r in circles:
        cv2.circle(visualization, (x, y), r, (0, 255, 0), 2)
    return visualization


class HoughCircleDetector:
    def __init__(self, table_mask):
        self.table_mask = table_mask

    def get_hough_circles(self, image):
        hough_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hough_img = hough_img * self.table_mask
        circles = cv2.HoughCircles(hough_img, cv2.HOUGH_GRADIENT, minDist=15, dp=1.6, param1=100, param2=27,
                                   minRadius=8, maxRadius=25)
        return circles[0] if circles is not None else np.array([], dtype=np.float32).reshape((0, 3))

    def get_regions(self, image, return_radius=False):
        regions = []
        m, n = image.shape[:2]
        for x, y, r in self.get_hough_circles(image):
            half_side = min(r * 1.5, n - x, x, m - y, y)
            center = x, y
            borders = int(x - half_side), int(x + half_side + 1), int(y - half_side), int(y + half_side + 1)
            regions.append((center, borders, r) if return_radius else (center, borders))
        return regions
