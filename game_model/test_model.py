from game_model.model import Board, BallType
import sympy.geometry as geom


def test_game_model():
    tests = [
        (
            8, 8,
            [geom.Point(0, 1), geom.Point(8, 0), geom.Point(8, 8), geom.Point(4, 9)],
            [(BallType.MAROON_STRIPED, geom.Point(3, 3)),
             (BallType.BLACK, geom.Point(4, 7)),
             (BallType.YELLOW_SOLID, geom.Point(6, 5))],
            [(BallType.MAROON_STRIPED, (2, 7)),
             (BallType.BLACK, (1, 3)),
             (BallType.YELLOW_SOLID, (5, 5))]
        ),
        (
            16, 16,
            [geom.Point(0, 0), geom.Point(8, 0), geom.Point(8, 8), geom.Point(0, 8)],
            [(BallType.WHITE, geom.Point(3, 3)),
              (BallType.BLACK, geom.Point(4, 7))],
            [(BallType.WHITE, (6, 10)),
             (BallType.BLACK, (8, 2))]
        ),
        (
            2, 4,
            [geom.Point(0, 0), geom.Point(3, 0), geom.Point(3, 3), geom.Point(0, 6)],
            [(BallType.WHITE, geom.Point(2, 2)),
             (BallType.BLACK, geom.Point(2, 1))],
            [(BallType.WHITE, (1, 2)),
             (BallType.BLACK, (1, 3))]
        ),
        (
            4, 4,
            [geom.Point(0, 0), geom.Point(6, 0), geom.Point(5, 3), geom.Point(2, 3)],
            [(BallType.WHITE, geom.Point(3, 2)),
              (BallType.BLACK, geom.Point(5, 3))],
            [(BallType.WHITE, (2, 2)),
             (BallType.BLACK, (4, 0))]
        ),
    ]
    for X, Y, table, balls, expected in tests:
        b = Board(table, X, Y)
        b.add_balls(balls)
        assert b.balls == expected
        b.clear_balls()
        assert len(b.balls) == 0
