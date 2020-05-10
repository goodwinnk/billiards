from enum import Enum
from typing import Optional, List, Tuple

import cv2
import numpy as np
from scipy.spatial import distance
from sympy import Point2D

from ball_detection import HoughCircleDetector, MotionDetector, BallRegion
from ball_detection import visualize_balls_on_image, BallDetector, BallType as BallDetectorBallType
from game_model.model import Board, BallType
from table_recognition.find_table_polygon import find_table_layout_on_frame
from table_recognition.highlight_table import highlight_table_on_frame
from hole_recognition.hole_nn_model import HoleDetector
from hole_recognition.process_holes import rotate_table


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
    def __init__(self, ball_detect_net_path: str,
                 hole_detect_net_path: str = None,
                 table_size_mm: Tuple[int, int] = (990, 1980),
                 ball_size_mm: int = 57):
        self.table_size_mm = table_size_mm
        self.ball_size_mm = ball_size_mm

        self.ball_detect_net_path = ball_detect_net_path
        self.ball_detector: Optional[BallDetector] = None

        self.hole_detector: Optional[HoleDetector] = None
        if hole_detect_net_path is not None:
            self.hole_detector = HoleDetector()
            self.hole_detector.load(hole_detect_net_path)

        # TODO: Create type for ball result with correspondent accessors
        self.balls: List[BallRegion] = []

        self.table_layout: Optional[np.ndarray] = None
        self.min_ball_radius_pixels: Optional[int] = None
        self.max_ball_radius_pixels: Optional[int] = None

        self.board_size = (400, 800)
        self.board: Optional[Board] = None

        self.log: List[VideoEvent] = []

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _need_to_update_board(self, frame, index):
        # TODO: need to understand that board should be updated
        return False

    def _update_balls_radius(self, table_layout):
        n = len(table_layout)
        side_lengths = [distance.euclidean(table_layout[i], table_layout[(i + 1) % n]) for i in range(n)]
        min_side = min(side_lengths)
        max_side = max(side_lengths)

        self.min_ball_radius_pixels = int(min_side / self.table_size_mm[1] * self.ball_size_mm)
        self.max_ball_radius_pixels = int(max_side / self.table_size_mm[0] * self.ball_size_mm)

    def _update_board(self, frame, index):
        un_oriented_table_layout = find_table_layout_on_frame(frame)
        if un_oriented_table_layout is None:
            self.board = None
            self.table_layout = None
            self.ball_detector = None
            return

        self.log.append(VideoEvent(VideoEvent.EventType.CAMERA_STABLE, index))
        if self.hole_detector is None:
            self.table_layout: np.ndarray = PoolCV.orient_table_for_model(un_oriented_table_layout)
        else:
            self.table_layout: np.array = rotate_table(self.hole_detector, frame, un_oriented_table_layout)
        self._update_balls_radius(self.table_layout)

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

    def draw_game_on_image(self, frame, draw_net=True):
        visualize_balls_on_image(frame, self.balls)
        highlight_table_on_frame(frame, self.table_layout)
        if draw_net:
            PoolCV.draw_net_on_frame(frame, self.min_ball_radius_pixels)

    def get_model_image(self):
        if self.board is None:
            return np.zeros(self.board_size + (3,), dtype=np.uint8)
        return self.board.to_image()

    # TODO: Use compatible types to make conversion redundant
    @staticmethod
    def convert_balls(balls) -> List[Tuple[BallType, Point2D]]:
        model_balls = []
        for ball in balls:
            ball_recognition_type = ball.region_type
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

            model_balls.append((ball_model_type, Point2D(int(ball.center.x), int(ball.center.y))))

        return model_balls

    @staticmethod
    def create_mask(image_shape, contour):
        binary_black = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        cv2.drawContours(binary_black, [contour], 0, 1, -1)
        return binary_black

    @staticmethod
    def draw_net_on_frame(frame, step):
        height, width, _ = frame.shape
        net_color = (60, 60, 60)
        for x in range(0, width, step):
            cv2.line(frame, (x, 0), (x, height), net_color, thickness=1)

        for y in range(0, height, step):
            cv2.line(frame, (0, y), (width, y), net_color, thickness=1)

    @staticmethod
    def orient_table_for_model(table_layout, ratio=2, precision=0.9):
        n = len(table_layout)
        assert n == 4

        y_edge_sum = [table_layout[i][1] + table_layout[(i + 1) % n][1] for i in range(n)]
        closest_to_bottom = max(y_edge_sum)
        index_closest_to_bottom = y_edge_sum.index(closest_to_bottom)

        edge_lengths = [distance.euclidean(table_layout[i], table_layout[(i + 1) % n]) for i in range(n)]

        bottom_top_lengths = edge_lengths[index_closest_to_bottom] + edge_lengths[(index_closest_to_bottom + 2) % n]
        side_lengths = edge_lengths[(index_closest_to_bottom + 1) % n] + edge_lengths[(index_closest_to_bottom + 3) % n]

        is_counter_clockwise = table_layout[index_closest_to_bottom][0] <= \
                               table_layout[(index_closest_to_bottom + 1) % n][0]

        if bottom_top_lengths / side_lengths > ratio * precision:
            # consider the closest edge to be a long rail
            if is_counter_clockwise:
                short_rail_bottom_left_corner_index = (index_closest_to_bottom + 1) % n
            else:
                short_rail_bottom_left_corner_index = index_closest_to_bottom
        else:
            # consider the closest edge to be a short rail
            if is_counter_clockwise:
                short_rail_bottom_left_corner_index = index_closest_to_bottom
            else:
                short_rail_bottom_left_corner_index = (index_closest_to_bottom + 1) % n

        if is_counter_clockwise:
            return np.roll(table_layout, -short_rail_bottom_left_corner_index, axis=0)
        else:
            return np.roll(table_layout[::-1], -(n - 1 - short_rail_bottom_left_corner_index), axis=0)
