import sympy.geometry as geom
from typing import Dict, List
from enum import Enum
import cv2
import numpy as np


class BallType(Enum):
    """
    The billiard balls colors enum.
    """
    YELLOW_SOLID = 1
    BLUE_SOLID = 2
    RED_SOLID = 3
    PURPLE_SOLID = 4
    ORANGE_SOLID = 5
    GREEN_SOLID = 6
    MAROON_SOLID = 7
    BLACK = 8
    YELLOW_STRIPED = 9
    BLUE_STRIPED = 10
    RED_STRIPED = 11
    PURPLE_STRIPED = 12
    ORANGE_STRIPED = 13
    GREEN_STRIPED = 14
    MAROON_STRIPED = 15
    WHITE = 16

    def is_striped(self):
        return 9 <= self.value <= 15

    def is_solid(self):
        return 1 <= self.value <= 7

    def label(self) -> str:
        return str(self.value) if self != BallType.WHITE else ""


class Colors:
    table_color = [50, 170, 50]
    white = [255, 255, 255]
    black = [0, 0, 0]
    brown = [20, 70, 140]

    __ball_colors = [black,
              [0, 255, 255],  # yellow
              [255, 0, 0],  # blue
              [0, 0, 255],  # red
              [150, 50, 90],  # purple
              [0, 150, 255],  # orange
              [0, 255, 0],  # green
              [0, 30, 150],  # maroon
              white]

    @staticmethod
    def main_ball_color(ball_type: BallType):
        return Colors.__ball_colors[ball_type.value % 8] \
            if ball_type != BallType.WHITE else Colors.__ball_colors[-1]


class Board:
    """
    A class to store the billiard table model by the table and balls coordinates on the image.
    Use constructor to initialize Board with the table coordinates,
    add_balls method to add the coordinates of the billiard balls,
    clear method to delete all added balls in the model.
    """

    def __init__(self, init_table: List[geom.Point2D], x=750, y=1000):
        """
        Initialize the class with 4 billiard table corners.
        :param init_table: 4 corner coordinates of the table on the image.
          They will be treated in the given order as the left bottom, right bottom, right top, left top model corners.
        :param x: the horizontal pixel shape of the model picture.
        :param y: the vertical pixel shape of the model picture.
        """
        self.img_sz = (x, y)
        self.balls = []

        if len(init_table) != 4:
            raise AttributeError('Table array should contain only 4 points: the billiard table corners.')
        self.A, self.B, self.C, self.D = tuple(init_table)
        self.P = self.Q = self.line = self.X1 = self.X2 = self.Y1 = self.Y2 = self.x_len = self.y_len = None

        i1 = geom.intersection(geom.Line2D(self.A, self.B), geom.Line2D(self.C, self.D))
        if len(i1) == 1:
            self.P = i1[0]
        i2 = geom.intersection(geom.Line2D(self.A, self.D), geom.Line2D(self.C, self.B))
        if len(i2) == 1:
            self.Q = i2[0]
        if self.P is not None and self.Q is not None:
            self.line = geom.Line2D(self.P, self.Q).parallel_line(self.B)
        elif self.P is not None and self.Q is None:
            self.line = geom.Line2D(self.B, self.C)
        elif self.P is None and self.Q is not None:
            self.line = geom.Line2D(self.A, self.B)

        if self.P is not None:
            self.Y1 = geom.intersection(geom.Line2D(self.D, self.C), self.line)[0]
            self.y_len = self.Y1.distance(self.B)
        if self.Q is not None:
            self.X1 = geom.intersection(geom.Line2D(self.D, self.A), self.line)[0]
            self.x_len = self.X1.distance(self.B)

    def add_balls(self, added_balls: Dict[BallType, geom.Point2D]):
        """
        Adds the given billiard balls coordinates to the table model.
        :param added_balls: the ball color (see class BallColor) with ball coordinate point on the image
          (which was used for table coordinates in the constructor).
        """
        for color, p in added_balls.items():
            x, y = self.get_rectangular_coordinates(p)
            self.balls.append((color, (x, y)))

    def clear_balls(self):
        """
        Deletes all the added balls from the model.
        """
        self.balls.clear()

    def get_rectangular_coordinates(self, pos: geom.Point2D):
        """
        Calculate the model coordinates of the pos point inside the init table.
        """

        def get_coeff(p, x, x_len, a, b, c, d):
            if p is not None:
                k = geom.intersection(geom.Line2D(p, pos), self.line)[0]
                return x.distance(k) / x_len
            k1 = a.distance(d)
            k2 = b.distance(c)
            ln = geom.Line2D(a, d).parallel_line(pos)
            g1 = geom.intersection(ln, geom.Line2D(a, b))[0]
            g2 = geom.intersection(ln, geom.Line2D(c, d))[0]
            k3 = g1.distance(g2)
            return k2 * (k3 - k1) / (k3 * (k2 - k1))

        if self.P is None and self.Q is None:
            l1 = geom.Line2D(self.A, self.D)
            l2 = geom.Line2D(self.C, self.D)
            x_coeff = pos.distance(l1) / self.B.distance(l1)
            y_coeff = pos.distance(l2) / self.B.distance(l2)
        else:
            x_coeff = get_coeff(self.Q, self.X1, self.x_len, self.A, self.B, self.C, self.D)
            y_coeff = get_coeff(self.P, self.Y1, self.y_len, self.D, self.A, self.B, self.C)
        x = round(x_coeff * self.img_sz[0])
        y = round(y_coeff * self.img_sz[1])
        return x, y

    def to_image(self):
        """
        Returns the model representation of the current Board state.
        :return: image (np.array) of shape (X, Y) with all the added balls.
        """
        image = np.float32([[Colors.table_color for _ in range(self.img_sz[0])] for _ in range(self.img_sz[1])])

        r = 20  # ball radius
        for x in [0, self.img_sz[0]]:
            for y in [0, self.img_sz[1] // 2, self.img_sz[1]]:
                cv2.circle(image, (x, y), 2 * r, Colors.brown, thickness=-1)

        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.0
        thickness = 2

        def get_text_start_point(center_point, text):
            center_point_x, center_point_y = center_point
            text_sz, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_sz_x, text_sz_y = text_sz
            return (center_point_x - text_sz_x // 2,
                    center_point_y + text_sz_y // 2)

        for ball, pos in self.balls:
            ball_color = Colors.main_ball_color(ball)
            new_pos = (pos[0], pos[1])
            if ball.is_striped():
                cv2.circle(image, new_pos, r, Colors.white, thickness=-1)
                cv2.circle(image, new_pos, r - 4, ball_color, thickness=-1)
            else:
                cv2.circle(image, new_pos, r, ball_color, thickness=-1)
            cv2.circle(image, new_pos, r // 2, Colors.white, thickness=-1)
            label = ball.label()
            cv2.putText(image, label, get_text_start_point(new_pos, label),
                        font, thickness=thickness, color=Colors.black, fontScale=font_scale)

        return image


if __name__ == '__main__':
    # These table coordinates form a rectangular for simplicity reasons,
    # nevertheless, it is possible for them to form any convex quadrangle.
    table = [geom.Point(0, 0), geom.Point(10, 0), geom.Point(10, 10), geom.Point(0, 10)]

    balls = {BallType.YELLOW_SOLID: geom.Point(5, 5),
             BallType.BLUE_SOLID: geom.Point(1, 1),
             BallType.RED_SOLID: geom.Point(7, 2),
             BallType.PURPLE_SOLID: geom.Point(2, 7),
             BallType.ORANGE_SOLID: geom.Point(7, 7),
             BallType.GREEN_SOLID: geom.Point(4, 5),
             BallType.MAROON_SOLID: geom.Point(5, 3),
             BallType.BLACK: geom.Point(9, 6),
             BallType.YELLOW_STRIPED: geom.Point(2, 8),
             BallType.BLUE_STRIPED: geom.Point(2, 4),
             BallType.RED_STRIPED: geom.Point(9, 9),
             BallType.PURPLE_STRIPED: geom.Point(3, 6),
             BallType.ORANGE_STRIPED: geom.Point(6, 3),
             BallType.GREEN_STRIPED: geom.Point(6, 8),
             BallType.MAROON_STRIPED: geom.Point(3, 4),
             BallType.WHITE: geom.Point(5, 9)}

    board = Board(table)
    board.add_balls(balls)

    img = board.to_image()
    cv2.imwrite('../data/sync/billiard_model_example.jpg', img)
