import cv2

from data_utils.utils import frame_by_frame_play
from ball_detection import BallDetector, visualize_balls
from ball_detection.candidate_generation_hough import HoughCircleDetector


WEIGHTS_PATH = 'ball_detection/candidate_classifier/weights.pt'
DATA_DIR = 'data/sync/resources/012'
VIDEO_PATH = DATA_DIR + '/video012.mp4'
TABLE_MASK_PATH = DATA_DIR + '/table_mask.png'


table_mask = cv2.imread(TABLE_MASK_PATH)[:, :, 0]
candidate_generator = HoughCircleDetector(table_mask)
detector = BallDetector(candidate_generator)


def highlight_balls(image, index):
    balls = detector.get_balls(image)
    return visualize_balls(image, balls)


frame_by_frame_play(VIDEO_PATH, frame_modifier=highlight_balls)
