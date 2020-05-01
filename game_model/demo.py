from game_model.model import Board, BallType, Colors
import sympy.geometry as geom
import cv2
import json
from table_recognition.highlight_table import highlight_table_on_frame


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
    img = cv2.imread('../data/sync/game_model_demo/image.jpg')
    sample_input = img.copy()
    highlight_table_on_frame(sample_input, table_corners)
    for ball, point in balls.items():
        cv2.circle(sample_input,
                   (point.x, point.y),
                   radius=20, color=Colors.main_ball_color(ball),
                   thickness=6 if ball.is_solid() else 2)
    cv2.imwrite("../data/sync/game_model_demo/sample_input.jpg", sample_input)

    # model demo
    a, b, c, d = tuple(table_corners)
    b = Board([make_point(a), make_point(b), make_point(c), make_point(d)])
    b.add_balls(balls)
    cv2.imwrite('../data/sync/game_model_demo/result.jpg', b.to_image())
