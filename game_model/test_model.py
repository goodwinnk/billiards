from game_model.model import Board, BallColor
import sympy.geometry as geom


def test_game_model():
    tests = [
        (
            8, 8,
            [geom.Point(0, 1), geom.Point(8, 0), geom.Point(8, 8), geom.Point(4, 9)],
            {BallColor.MAROON_STRIPED: geom.Point(3, 3),
             BallColor.BLACK: geom.Point(4, 7),
             BallColor.YELLOW_SOLID: geom.Point(6, 5)},
            {(BallColor.MAROON_STRIPED, (2, 7)),
             (BallColor.BLACK, (1, 3)),
             (BallColor.YELLOW_SOLID, (5, 5))}
        ),
        (
            16, 16,
            [geom.Point(0, 0), geom.Point(8, 0), geom.Point(8, 8), geom.Point(0, 8)],
            {BallColor.WHITE: geom.Point(3, 3),
             BallColor.BLACK: geom.Point(4, 7)},
            {(BallColor.WHITE, (6, 10)),
             (BallColor.BLACK, (8, 2))}
        ),
        (
            2, 4,
            [geom.Point(0, 0), geom.Point(3, 0), geom.Point(3, 3), geom.Point(0, 6)],
            {BallColor.WHITE: geom.Point(2, 2),
             BallColor.BLACK: geom.Point(2, 1)},
            {(BallColor.WHITE, (1, 2)),
             (BallColor.BLACK, (1, 3))}
        ),
        (
            4, 4,
            [geom.Point(0, 0), geom.Point(6, 0), geom.Point(5, 3), geom.Point(2, 3)],
            {BallColor.WHITE: geom.Point(3, 2),
             BallColor.BLACK: geom.Point(5, 3)},
            {(BallColor.WHITE, (2, 2)),
             (BallColor.BLACK, (4, 0))}
        ),
    ]
    for X, Y, table, balls, expected in tests:
        b = Board(table, X, Y)
        b.add_balls(balls)
        assert set(b.balls) == expected
        b.clear_balls()
        assert len(b.balls) == 0
