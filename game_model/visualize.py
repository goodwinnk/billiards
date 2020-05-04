from typing import List, Tuple

import cv2
from sympy import geometry as geom

from game_model.model import BallType, Colors
from table_recognition.highlight_table import highlight_table_on_frame


def draw_recognition_data(frame, table_corners: List[List[int]], balls: List[Tuple[BallType, geom.Point2D]]):
    highlight_table_on_frame(frame, table_corners)
    for ball_type, point in balls:
        cv2.circle(frame,
                   (point.x, point.y),
                   radius=20, color=Colors.main_ball_color(ball_type),
                   thickness=6 if ball_type.is_solid() else 2)