import numpy as np
import cv2

from data_utils.utils import frame_by_frame_play
from ball_detection import BallDetector, visualize_balls, get_unique_balls
from ball_detection.candidate_generation_hough import HoughCircleDetector
from ball_detection.candidate_generation_motion import MotionDetector, get_background
from table_recognition.find_table_polygon import calc_mask, read_frames


# Project directory root should be used as working directory

WEIGHTS_PATH = 'ball_detection/candidate_classifier/weights.pt'
VIDEO_PATH = 'data/sync/poolcv_demo/20200120_115235-resized.m4v'
FRAMES_FOR_MASK = 20
FRAMES_FOR_BACKGROUND = 10

video = cv2.VideoCapture(VIDEO_PATH)
table_mask = calc_mask(video, FRAMES_FOR_MASK).astype(np.uint8)
background = get_background(table_mask.astype(np.bool), read_frames(video, FRAMES_FOR_BACKGROUND))
candidate_generators = [
    HoughCircleDetector(table_mask),
    MotionDetector(table_mask.astype(np.bool), background)
]
detector = BallDetector(candidate_generators, net_path=WEIGHTS_PATH)


def highlight_balls(image, _):
    balls = detector.get_balls(image)
    balls = get_unique_balls(balls)
    return visualize_balls(image, balls)


frame_by_frame_play(VIDEO_PATH, frame_modifier=highlight_balls)
