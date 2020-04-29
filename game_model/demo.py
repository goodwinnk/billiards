from game_model import Board, BallColor
import sympy.geometry as geom
import cv2
import json

if __name__ == '__main__':
    img = cv2.imread('../data/sync/game_model_demo/image.jpg')
    with open('../data/sync/game_model_demo/coordinates.json', 'r') as f:
        text = ''.join(f.readlines())
        data = json.loads(text)
    a, b, c, d = tuple(data['table'])


    def make_point(o):
        x, y = tuple(o)
        return geom.Point2D(x, y)


    b = Board([make_point(a), make_point(b), make_point(c), make_point(d)])
    switcher = {
        'yellow_solid': BallColor.YELLOW_SOLID,
        'blue_solid': BallColor.BLUE_SOLID,
        'red_solid': BallColor.RED_SOLID,
        'purple_solid': BallColor.PURPLE_SOLID,
        'orange_solid': BallColor.ORANGE_SOLID,
        'green_solid': BallColor.GREEN_SOLID,
        'maroon_solid': BallColor.MAROON_SOLID,
        'black': BallColor.BLACK,
        'yellow_striped': BallColor.YELLOW_STRIPED,
        'blue_striped': BallColor.BLUE_STRIPED,
        'red_striped': BallColor.RED_STRIPED,
        'purple_striped': BallColor.PURPLE_STRIPED,
        'orange_striped': BallColor.ORANGE_STRIPED,
        'green_striped': BallColor.GREEN_STRIPED,
        'maroon_striped': BallColor.MAROON_STRIPED,
        'white': BallColor.WHITE,
    }
    balls = {}
    for key, val in data.items():
        if key != 'table':
            x, y = tuple(val)
            balls[switcher[key]] = geom.Point2D(x, y)
    b.add_balls(balls)

    cv2.imwrite('../data/sync/game_model_demo/result.jpg', b.to_image())