from typing import List, Dict

import cv2
from sympy import geometry as geom

from game_model.model import BallType, Colors
from table_recognition.highlight_table import highlight_table_on_frame


def draw_recognition_data(frame, table_corners: List[List[int]], balls: Dict[BallType, geom.Point2D]):
    highlight_table_on_frame(frame, table_corners)
    for ball, point in balls.items():
        cv2.circle(frame,
                   (point.x, point.y),
                   radius=20, color=Colors.main_ball_color(ball),
                   thickness=6 if ball.is_solid() else 2)