import sympy.geometry as geom
from typing import Dict, List
from enum import Enum
import cv2
import numpy as np
import math


class BallColor(Enum):
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
        if len(init_table) != 4:
            raise AttributeError('Table array should contain only 4 points: the billiard table corners.')
        self.init_table = init_table
        self.X = x
        self.Y = y
        self.balls = []

    def add_balls(self, added_balls: Dict[BallColor, geom.Point2D]):
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
        def cross_product(p1: geom.Point2D, p2: geom.Point2D):
            return p1[0] * p2[1] - p1[1] * p2[0]

        def get_x_coeff(a, b, c, d, o):
            k0 = cross_product(a - o, d - o)
            k1 = cross_product(b - a, d - o) + cross_product(a - o, c - d)
            k2 = cross_product(b - a, c - d)
            # k0 + k1 * x + k2 * x**2 = 0
            if k2 == 0:
                return -k0 / k1
            ed = math.sqrt(k1 ** 2 - 4 * k0 * k2)
            x1 = (-k1 + ed) / (2 * k2)
            if 0 <= x1 <= 1:
                return x1
            x2 = (-k1 - ed) / (2 * k2)
            return x2

        a, b, c, d = tuple(self.init_table)
        x = round(get_x_coeff(a, b, c, d, pos) * self.X)
        y = round(get_x_coeff(a, d, c, b, pos) * self.Y)
        return geom.Point(x, y)

    def to_image(self):
        """
        Returns the model representation of the current Board state.
        :return: image (np.array) of shape (X, Y) with all the added balls.
        """
        table_color = [50, 170, 50]
        white = [255, 255, 255]
        black = [0, 0, 0]
        image = np.float32([[table_color for _ in range(self.X)] for _ in range(self.Y)])
        colors = [black,
                  [0, 255, 255],   # yellow
                  [255, 0, 0],     # blue
                  [0, 0, 255],     # red
                  [150, 50, 90],   # purple
                  [0, 150, 255],   # orange
                  [0, 255, 0],     # green
                  [0, 30, 150],    # maroon
                  white]

        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.0
        thickness = 2

        def get_text_start_point(center_point, text):
            center_point_x, center_point_y = center_point
            text_sz, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_sz_x, text_sz_y = text_sz
            return (center_point_x - text_sz_x // 2,
                    center_point_y + text_sz_y // 2)

        for color, pos in self.balls:
            v = color.value
            c = colors[v % 8] if v != 16 else colors[-1]
            r = 20
            new_pos = (pos[0], self.Y - pos[1])
            if 9 <= v <= 15:
                cv2.circle(image, new_pos, r, white, thickness=-1)
                cv2.circle(image, new_pos, r - 4, c, thickness=-1)
            else:
                cv2.circle(image, new_pos, r, c, thickness=-1)
            cv2.circle(image, new_pos, r // 2, white, thickness=-1)
            label = str(v) if v != 16 else ""
            cv2.putText(image, label, get_text_start_point(new_pos, label),
                        font, thickness=thickness, color=black, fontScale=font_scale)

        return image


if __name__ == '__main__':
    table = [geom.Point(0, 0), geom.Point(10, 0), geom.Point(10, 10), geom.Point(0, 10)]
    balls = {BallColor.YELLOW_SOLID: geom.Point(5, 5),
             BallColor.BLUE_SOLID: geom.Point(1, 1),
             BallColor.RED_SOLID: geom.Point(7, 2),
             BallColor.PURPLE_SOLID: geom.Point(2, 7),
             BallColor.ORANGE_SOLID: geom.Point(7, 7),
             BallColor.GREEN_SOLID: geom.Point(4, 5),
             BallColor.MAROON_SOLID: geom.Point(5, 3),
             BallColor.BLACK: geom.Point(9, 6),
             BallColor.YELLOW_STRIPED: geom.Point(2, 8),
             BallColor.BLUE_STRIPED: geom.Point(2, 4),
             BallColor.RED_STRIPED: geom.Point(9, 9),
             BallColor.PURPLE_STRIPED: geom.Point(3, 6),
             BallColor.ORANGE_STRIPED: geom.Point(6, 3),
             BallColor.GREEN_STRIPED: geom.Point(6, 8),
             BallColor.MAROON_STRIPED: geom.Point(3, 4),
             BallColor.WHITE: geom.Point(5, 9)}

    board = Board(table)
    board.add_balls(balls)

    img = board.to_image()
    cv2.imwrite('../data/sync/billiard_model_example.jpg', img)
