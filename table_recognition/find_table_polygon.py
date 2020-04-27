import argparse
import math
from collections import deque
from copy import deepcopy
from typing import Iterable, Tuple

import cv2
import numpy as np
from scipy.spatial import ConvexHull


FRAMES_TO_DETECT_TABLE = 40


# finds mask for convex polygon on the image of given size
def find_convex_hull_mask(size, hull):
    n, m = size
    mask = np.zeros((n, m))

    i = 0
    I = 0
    # finds lefties and rightest points in the hull
    for q, (y, x) in enumerate(hull):
        if x < hull[i][1]:
            i = q
        if x > hull[I][1]:
            I = q
    j = i
    lx = hull[i][1]
    rx = hull[I][1]

    def get_prev(_id):
        return (_id - 1) if (_id > 0) else (len(hull) - 1)

    def get_next(_id):
        return (_id + 1) % len(hull)

    def get_y_on_segment(p1, p2, x, vertical_segment_policy: str = 'up'):
        y1, x1 = p1
        y2, x2 = p2
        if x1 > x2 or (x1 == x2 and y1 > y2):
            x1, y1, x2, y2, p1, p2 = x2, y2, x1, y1, p2, p1  # swap
        if x1 == x2:
            return y1 if vertical_segment_policy == 'down' else y2
        alpha = (x - x1) / (x2 - x1)
        return int(np.round((1 - alpha) * y1 + alpha * y2))

    lx = max(0, lx)
    rx = min(n - 1, rx)

    for x in range(lx, rx + 1):
        while hull[i][1] < x:
            i = get_prev(i)
        while hull[j][1] < x:
            j = get_next(j)
        yi = get_y_on_segment(hull[i], hull[get_next(i)], x, 'up')
        yj = get_y_on_segment(hull[j], hull[get_prev(j)], x, 'down')
        yi = max(0, min(m - 1, yi))
        yj = max(0, min(m - 1, yj))
        if yi > yj:
            yi, yj = yj, yi
        mask[x, yi: yj + 1] = 1

    return mask


def parse_args():
    parser = argparse.ArgumentParser(
        description='Finds table polygon for each frame'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to the video to be processed'
    )
    parser.add_argument(
        '--layout',
        type=str,
        required=True,
        help='Path to the output fill for storing layout'
    )
    return parser.parse_args()


# finds mask of pixels which are close enough to the mean color of the table
def find_table_mask(
        frame,
        same_color_threshold: int = 4000,
        table_color_center_ratio_size: float = 0.1
):
    n, m, _ = frame.shape

    skip_pxl_n = int(n * (1 - table_color_center_ratio_size) / 2)
    skip_pxl_m = int(m * (1 - table_color_center_ratio_size) / 2)

    table_color = np.mean(frame[skip_pxl_n: n - skip_pxl_n, skip_pxl_m: m - skip_pxl_m, :], axis=(0, 1)) \
        .astype(dtype=int).clip(0, 255)

    return (np.sum((frame - table_color) ** 2, axis=2) < same_color_threshold).astype(int)


# takes 2d array of bools, finds connected component of Trues with the largest area and leaves only that component
# so other component will be False
# PS. Two pixels are connected iff they have commond side
def find_largest_area_component_mask(mask, center_ratio_size: float = 0.5):
    q = deque()
    n, m = mask.shape

    skip_pxl_n = int(n * (1 - center_ratio_size) / 2)
    skip_pxl_m = int(m * (1 - center_ratio_size) / 2)

    color = np.zeros_like(mask)
    current_color = 0
    best_component_size, best_component_color = 0, 0
    for si in range(skip_pxl_n, n - skip_pxl_n):
        for sj in range(skip_pxl_m, m - skip_pxl_m):
            if not mask[si, sj] or color[si, sj] != 0:
                continue
            current_color += 1
            component_size = 0
            q.clear()
            q.append((si, sj))

            while len(q) > 0:
                vi, vj = q.pop()
                if vi < 0 or vi >= n or vj < 0 or vj >= m or color[vi, vj] != 0 or not mask[vi, vj]:
                    continue
                color[vi, vj] = current_color
                component_size += 1
                q.append((vi - 1, vj))
                q.append((vi + 1, vj))
                q.append((vi, vj - 1))
                q.append((vi, vj + 1))

            if component_size > best_component_size:
                best_component_size, best_component_color = component_size, current_color
    return color == best_component_color


def find_table_polygon(
        frame,
        largest_component_center_ratio_size: int = 0.5
):
    frame = frame[:, :, ::-1]

    mask = find_table_mask(frame)
    mask = find_largest_area_component_mask(mask, largest_component_center_ratio_size)

    masked_frame = deepcopy(frame)
    masked_frame[mask == True] = [255, 255, 255]
    masked_frame[mask == False] = [0, 0, 0]

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_line_length = 200
    max_line_gap = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, min_line_length, max_line_gap)

    if lines is None or len(lines) < 3:
        return None

    points = np.array([(0, 0) for _ in range(2 * len(lines[:, 0, :]))])

    for i, (x1, y1, x2, y2) in enumerate(lines[:, 0, :]):
        points[2 * i] = (x1, y1)
        points[2 * i + 1] = (x2, y2)

    return points[ConvexHull(points).vertices]


# removes vertex which produces the largest angle until it greater than threshold
def remove_big_angles_from_hull(hull, angle_threshold=np.pi * 160 / 180, min_sides=4):
    n = len(hull)

    def get_angle(i):
        prv = (i - 1 + n) % n
        nxt = (i + 1) % n

        v1 = hull[nxt] - hull[i]
        v2 = hull[prv] - hull[i]

        return math.acos(np.sum(v1 * v2) / np.linalg.norm(v1) / np.linalg.norm(v2))

    while len(hull) > min_sides:
        mx_angle_id = 0
        mx_angle = 0
        for i in range(n):
            ith_angle = get_angle(i)
            if ith_angle > mx_angle:
                mx_angle, mx_angle_id = ith_angle, i
        if mx_angle < angle_threshold:
            break
        hull = np.concatenate((hull[: mx_angle_id], hull[mx_angle_id + 1: ]))
        n -= 1

    return hull


def intersect_lines(line1, line2):
    p1, p2 = line1
    q1, q2 = line2

    pv = p2 - p1
    qv = q2 - q1

    denum = pv[1] * qv[0] - pv[0] * qv[1]
    assert math.fabs(denum) > np.float64(1e-9)

    r = q1 - p1
    num = r[1] * pv[0] - r[0] * pv[1]

    t = num / denum

    return q1 + t * qv


def take_longest_sides_from_hull(hull, k):
    n = len(hull)
    assert n >= k
    ids = list(range(n))
    ids.sort(key=lambda i: np.linalg.norm(hull[i] - hull[(i + 1) % n]), reverse=True)
    ids = list(sorted(ids[: k]))

    khull = []
    for it in range(k):
        i1 = ids[it]
        i2 = (i1 + 1) % n
        j1 = ids[(it + 1) % k]
        j2 = (j1 + 1) % n

        khull.append(intersect_lines((hull[i1], hull[i2]), (hull[j1], hull[j2])))

    return np.array(khull, dtype=int)


# Takes video, finds tables polygon vertex coordinates for each frame and saves layout
def find_table_layout(input_video_path, layout_path):
    capture = cv2.VideoCapture(input_video_path)

    with open(layout_path, 'w') as layout_file:
        while capture.isOpened():
            response, frame = capture.read()

            if not response:
                break

            hull = find_table_polygon(deepcopy(frame))
            hull = remove_big_angles_from_hull(hull)
            hull = take_longest_sides_from_hull(hull, 4)
            assert len(hull) == 4
            hull = hull[:, ::-1]

            for x, y in hull:
                layout_file.write(f'{x} {y} ')
            layout_file.write('\n')


# Calculates hulls for provides frames
def get_table_hulls(frames: Iterable[np.array], resolution: Tuple[int]):
    center = np.array(resolution)[::-1] // 2
    layout = []

    for frame in frames:
        hull = find_table_polygon(deepcopy(frame))
        if hull is None:
            layout.append(np.zeros((4, 2), dtype=np.int32))
            continue

        hull = remove_big_angles_from_hull(hull)
        hull = take_longest_sides_from_hull(hull, 4)

        center_vectors = hull - center
        angles = np.arctan2(center_vectors[:, 0], center_vectors[:, 1])
        hull_point_indices = np.argsort(angles)
        hull = hull[hull_point_indices]

        layout.append(hull)
    return np.array(layout, dtype=np.int32)


# Reads FRAMES_TO_DETECT_TABLE from the provided video capture
# Changes the CAP_PROP_POS_FRAMES position in the video capture
def read_frames(video: cv2.VideoCapture):
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = int(frame_count / FRAMES_TO_DETECT_TABLE)
    for frame_number in range(0, frame_count, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        response, frame = video.read()
        yield frame


# Calculates FRAMES_TO_DETECT_TABLE table hulls for the provided video and averages the good ones.
# Currently hulls with negative coordinates are discarded.
# Returns a table mask
def calc_mask(video: cv2.VideoCapture):
    resolution = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    hulls = get_table_hulls(read_frames(video), resolution)
    relevant_hulls = hulls.all(axis=(1, 2))
    mean_hull = hulls[relevant_hulls].mean(axis=0).astype(np.int32)
    return find_convex_hull_mask(resolution, mean_hull)


if __name__ == '__main__':
    np.random.seed(42)
    args = parse_args()
    find_table_layout(args.video, args.layout)
