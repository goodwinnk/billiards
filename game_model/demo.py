import json

import cv2
import sympy.geometry as geom

from game_model.model import Board, BallType
from game_model.visualize import draw_recognition_data


def make_point(o):
    x, y = tuple(o)
    return geom.Point2D(x, y)


if __name__ == '__main__':
    with open('../data/sync/game_model_demo/coordinates.json', 'r') as f:
        text = ''.join(f.readlines())
        data = json.loads(text)

    table_corners = data['table']
    balls = {}
    for key, val in data["balls"].items():
        balls[BallType[key.upper()]] = make_point(val)

    # sample image
    sample_input = cv2.imread('../data/sync/game_model_demo/image.jpg').copy()
    draw_recognition_data(sample_input, table_corners, balls)
    cv2.imwrite("../data/sync/game_model_demo/sample_input.jpg", sample_input)

    # model demo
    a, b, c, d = tuple(table_corners)
    b = Board([make_point(a), make_point(b), make_point(c), make_point(d)])
    b.add_balls(balls)
    cv2.imwrite('../data/sync/game_model_demo/result.jpg', b.to_image())
