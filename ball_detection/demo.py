import numpy as np
import cv2

from data_utils.utils import frame_by_frame_play
from ball_detection import BallDetector, visualize_balls
from ball_detection.candidate_generation_hough import HoughCircleDetector
from ball_detection.candidate_generation_motion import MotionDetector


WEIGHTS_PATH = 'ball_detection/candidate_classifier/weights.pt'
DATA_DIR = 'data/sync/resources/010'
VIDEO_PATH = DATA_DIR + '/video010.mp4'
TABLE_MASK_PATH = DATA_DIR + '/table_mask.png'
BACKGROUND_PATH = DATA_DIR + '/background.png'


table_mask = cv2.imread(TABLE_MASK_PATH)[:, :, 0]
background = cv2.imread(BACKGROUND_PATH)
candidate_generators = [
    HoughCircleDetector(table_mask),
    MotionDetector(table_mask.astype(np.bool), background)
]
detector = BallDetector(candidate_generators)


def highlight_balls(image, index):
    balls = detector.get_balls(image)
    return visualize_balls(image, balls)


frame_by_frame_play(VIDEO_PATH, frame_modifier=highlight_balls)
