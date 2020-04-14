import numpy as np
import cv2


def get_hough_circles(image, table_mask):
    hough_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hough_img = hough_img * table_mask
    circles = cv2.HoughCircles(hough_img, cv2.HOUGH_GRADIENT, minDist=15, dp=1.6, param1=100, param2=27,
                               minRadius=8, maxRadius=25)
    return circles[0] if circles is not None else np.array([], dtype=np.float32).reshape((0, 3))


def visualize_circles(image, circles):
    visualization = image.copy()
    if not len(circles):
        return visualization
    for x, y, r in circles:
        cv2.circle(visualization, (x, y), r, (0, 255, 0), 2)
    return visualization


def get_circle_region_borders(image, circles):
    cuts = []
    m, n = image.shape[:2]
    for x, y, r in circles:
        half_side = min(r * 1.5, n - x, x, m - y, y)
        center = x, y
        borders = int(x - half_side), int(x + half_side + 1), int(y - half_side), int(y + half_side + 1)
        cuts.append((center, r, borders))
    return cuts
