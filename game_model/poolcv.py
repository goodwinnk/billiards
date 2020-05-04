from enum import Enum
from typing import Optional, List, Tuple

import cv2
import numpy as np
from sympy import Point2D

from ball_detection.ball_detector import visualize_balls_on_image, BallDetector, BallType as BallDetectorBallType
from ball_detection.candidate_generation_hough import HoughCircleDetector
from ball_detection.candidate_generation_motion import MotionDetector
from game_model.model import Board, BallType
from table_recognition.find_table_polygon import find_table_layout_on_frame
from table_recognition.highlight_table import highlight_table_on_frame


class VideoEvent:
    class EventType(Enum):
        GAME_STARTED = 1  # TODO: Detect original position
        GAME_ENDED = 2  # TODO: Detect game has ended
        SHOT_STARTED = 3  # TODO
        SHOT_ENDED = 4  # TODO
        BALL_IN_THE_POCKET = 3  # TODO
        CAMERA_CHANGED = 4  # TODO
        CAMERA_STABLE = 5

    def __init__(self, event_type: EventType, frame_index: int):
        self.type = event_type
        self.frame_index = frame_index


class PoolCV:
    def __init__(self, ball_detect_net_path: str):
        self.ball_detect_net_path = ball_detect_net_path
        self.ball_detector: Optional[BallDetector] = None

        # TODO: Create type for ball result with correspondent accessors
        self.balls: List[Tuple[
            Tuple[float, float],  # center
            Tuple[int, int, int, int],  # region
            BallDetectorBallType
        ]] = []

        self.table_layout: Optional[np.ndarray] = None

        self.board_size = (400, 800)
        self.board: Optional[Board] = None

        self.log: List[VideoEvent] = []

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _need_to_update_board(self, frame, index):
        # TODO: need to understand that board should be updated
        return False

    def _update_board(self, frame, index):
        self.log.append(VideoEvent(VideoEvent.EventType.CAMERA_STABLE, index))
        self.table_layout: np.ndarray = find_table_layout_on_frame(frame)

        # TODO: need to have correct order of corners
        self.board = Board(list(map(lambda corner: Point2D(corner), self.table_layout)),
                           x=self.board_size[0], y=self.board_size[1])

        # TODO: get table_mask and table_background images
        table_mask = PoolCV.create_mask(frame.shape, self.table_layout)
        table_background = cv2.bitwise_and(frame, frame, mask=table_mask)

        self.ball_detector = BallDetector(
            candidate_generators=[
                HoughCircleDetector(table_mask),
                MotionDetector(table_mask.astype(np.bool), table_background)
            ],
            net_path=self.ball_detect_net_path)

    def update(self, frame, index):
        if self.board is None or self._need_to_update_board(frame, index):
            self._update_board(frame, index)

        if self.board is None:
            return

        if self.ball_detector is not None:
            self.balls = self.ball_detector.get_balls(frame)

            # TODO: Track the balls and update only relevant
            # TODO: Use size of balls for tracking
            # TODO: Clean multiple balls on overlapping positions
            self.board.clear_balls()
            self.board.add_balls(PoolCV.convert_balls(self.balls))

    def draw_game_on_image(self, frame):
        visualize_balls_on_image(frame, self.balls)
        highlight_table_on_frame(frame, self.table_layout)

    def get_model_image(self):
        if self.board is None:
            return np.zeros(self.board_size + (3,), dtype=np.uint8)
        return self.board.to_image()

    # TODO: Use compatible types to make conversion redundant
    @staticmethod
    def convert_balls(balls) -> List[Tuple[BallType, Point2D]]:
        model_balls = []
        for center, region, ball_recognition_type in balls:
            if ball_recognition_type == BallDetectorBallType.INTEGRAL:
                ball_model_type = BallType.GENERAL_SOLID
            elif ball_recognition_type == BallDetectorBallType.STRIPED:
                ball_model_type = BallType.GENERAL_STRIPED
            elif ball_recognition_type == BallDetectorBallType.WHITE:
                ball_model_type = BallType.WHITE
            elif ball_recognition_type == BallDetectorBallType.BLACK:
                ball_model_type = BallType.BLACK
            elif ball_recognition_type == BallDetectorBallType.FALSE:
                continue
            else:
                raise ValueError("Unknown value")

            model_balls.append((ball_model_type, Point2D(int(center[0]), int(center[1]))))

        return model_balls

    @staticmethod
    def create_mask(image_shape, contour):
        binary_black = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        cv2.drawContours(binary_black, [contour], 0, 1, -1)
        return binary_black
