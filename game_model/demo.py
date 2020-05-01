import game_model as gm
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


    b = gm.Board([make_point(a), make_point(b), make_point(c), make_point(d)])
    switcher = {
        'yellow_solid': gm.BallColor.YELLOW_SOLID,
        'blue_solid': gm.BallColor.BLUE_SOLID,
        'red_solid': gm.BallColor.RED_SOLID,
        'purple_solid': gm.BallColor.PURPLE_SOLID,
        'orange_solid': gm.BallColor.ORANGE_SOLID,
        'green_solid': gm.BallColor.GREEN_SOLID,
        'maroon_solid': gm.BallColor.MAROON_SOLID,
        'black': gm.BallColor.BLACK,
        'yellow_striped': gm.BallColor.YELLOW_STRIPED,
        'blue_striped': gm.BallColor.BLUE_STRIPED,
        'red_striped': gm.BallColor.RED_STRIPED,
        'purple_striped': gm.BallColor.PURPLE_STRIPED,
        'orange_striped': gm.BallColor.ORANGE_STRIPED,
        'green_striped': gm.BallColor.GREEN_STRIPED,
        'maroon_striped': gm.BallColor.MAROON_STRIPED,
        'white': gm.BallColor.WHITE,
    }
    balls = {}
    for key, val in data.items():
        if key != 'table':
            x, y = tuple(val)
            balls[switcher[key]] = geom.Point2D(x, y)
    b.add_balls(balls)

    cv2.imwrite('../data/sync/game_model_demo/result.jpg', b.to_image())