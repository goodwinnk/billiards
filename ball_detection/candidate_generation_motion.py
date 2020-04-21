import numpy as np
import cv2


MIN_BALL_CONTOUR_LEN = 40
MAX_BALL_CONTOUR_LEN = 200
BLUR_PARAMETERS = [(15, 15), 0]


def get_background(table_mask, frames):
    index_table_mask = np.stack([table_mask] * 3, axis=2)
    seq_frames = []

    for frame in frames:
        frame = cv2.GaussianBlur(frame, *BLUR_PARAMETERS)
        frame_table = frame[index_table_mask]
        seq_frames.append(frame_table)

    median_table_img = np.zeros((table_mask.shape + (3,)), dtype=np.uint8)
    median_table_img[index_table_mask] = np.median(np.array(seq_frames), axis=0)
    return median_table_img


class MotionDetector:
    def __init__(self, table_mask, background):
        self.table_mask = table_mask
        self.background = background

    def get_color_deviation(self, image):
        image = cv2.GaussianBlur(image, *BLUR_PARAMETERS)
        diff = cv2.absdiff(image, self.background)
        deviation = np.zeros(image.shape[:2], dtype=np.int16)
        deviation[self.table_mask] = diff[self.table_mask].sum(axis=1)
        return deviation

    def get_motion_mask(self, image):
        color_deviation = self.get_color_deviation(image)
        discrete_color_deviation = np.uint8(color_deviation * 255 / color_deviation.max())
        return cv2.adaptiveThreshold(discrete_color_deviation, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 81, -50)

    def get_regions(self, image):
        motion_mask = self.get_motion_mask(image)
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        m, n = image.shape[:2]
        regions = []
        for contour in contours:
            if MIN_BALL_CONTOUR_LEN <= len(contour) <= MAX_BALL_CONTOUR_LEN:
                cx, cy = center = tuple(map(int, contour.mean(axis=0)[0]))
                xs, ys = contour.squeeze().T
                max_coord_delta = max(np.abs(ys - cy).max(), np.abs(xs - cx).max())
                half_side = min(int(max_coord_delta * 1.5), n - cx, cx, m - cy, cy)
                borders = cx - half_side, cx + half_side + 1, cy - half_side, cy + half_side + 1
                regions.append((center, borders))
        return regions
