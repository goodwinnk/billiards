from enum import Enum
from collections import namedtuple
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union, Tuple

import numpy as np


class BallType(Enum):
    FALSE = 0
    INTEGRAL = 1
    STRIPED = 2
    WHITE = 3
    BLACK = 4


Point = namedtuple('Point', 'x y')
Rectangle = namedtuple('Rectangle', 'x0 y0 x1 y1')
BallRegion = namedtuple('BallRegion', 'center box region_type')


def get_center(rectangle: Rectangle) -> Point:
    return Point(int((rectangle.x0 + rectangle.x1) / 2), int((rectangle.y0 + rectangle.y1) / 2))


class BallCandidate:
    __slots__ = ['center', 'box', 'radius']

    def __init__(self, center: Point, box: Rectangle, radius: Optional[Union[float, int]] = None):
        self.center = center
        self.box = box
        self.radius = radius

    def __iter__(self):
        yield self.center
        yield self.box
        yield self.radius


class CandidateGenerator(metaclass=ABCMeta):
    @abstractmethod
    def get_regions(self, image: np.ndarray) -> List[BallCandidate]:
        pass


CANDIDATE_PADDING_COEFFICIENT = 1.5


def area(r: Rectangle) -> Union[int, float]:
    return (r.x1 - r.x0) * (r.y1 - r.y0)


def intersect(r1: Rectangle, r2: Rectangle) -> Rectangle:
    return Rectangle(max(r1.x0, r2.x0), max(r1.y0, r2.y0), min(r1.x1, r2.x1), min(r1.y1, r2.y1))


def valid_rectangle(r: Rectangle) -> bool:
    return (r.x0 < r.x1) and (r.y0 < r.y1)


def iou(ball1: Rectangle, ball2: Rectangle) -> float:
    intersection = intersect(ball1, ball2)
    if not valid_rectangle(intersection):
        return 0.
    a1, a2 = area(ball1), area(ball2)
    ai = area(intersection)
    au = a1 + a2 - ai
    return ai / au


def fit_region_in_image(radius: Union[int, float], center: Point, image_resolution: Tuple[int]) -> Rectangle:
    m, n = image_resolution
    adjusted_radius = radius * CANDIDATE_PADDING_COEFFICIENT
    half_side = min(adjusted_radius, n - center.x, center.x, m - center.y, center.y)
    return Rectangle(int(center.x - half_side), int(center.y - half_side),
                     int(center.x + half_side + 1), int(center.y + half_side + 1))
