from hmmlearn import hmm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from copy import deepcopy
from collections import deque

from src.video_operations import save_frames_as_video
from scipy.spatial import ConvexHull

np.random.seed(42)


# assume that table has blue color and table cover the biggest blue area
# cell is blue iff (B - R > threshold) & (B - G > threshold)
def blue_table_mask(frame: np.ndarray, threshold: int = 25):
    R = frame[:, :, 0].astype(int)
    G = frame[:, :, 1].astype(int)
    B = frame[:, :, 2].astype(int)
    return (B - R > threshold) & (B - G > threshold)


# takes video, masks blue parts, saves generated two-colored video
def exp1(input_video_path: str, output_video_path: str, verbose=True):
    capture = cv2.VideoCapture(input_video_path)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    masked_frames = []

    while capture.isOpened():
        response, frame = capture.read()
        if not response:
            break
        frame = frame[:, :, ::-1]  # RGB
        mask = blue_table_mask(frame).astype(dtype='uint8')

        masked_frame = frame
        masked_frame[mask == True] = [255, 255, 255]
        masked_frame[mask == False] = [0, 0, 0]

        masked_frames.append(masked_frame)

        if verbose:
            print(len(masked_frames))

    save_frames_as_video(output_video_path, masked_frames, fps)


# takes video, applies blue table mask, applies Canny, applies Hough transform, draws Hugh lines
def exp2(input_video_path: str, output_video_path: str, verbose=True):
    capture = cv2.VideoCapture(input_video_path)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    imgs = []

    while capture.isOpened():
        response, frame = capture.read()

        if not response:
            break

        frame = frame[:, :, ::-1]

        mask = blue_table_mask(frame).astype('int')

        masked_frame = deepcopy(frame)
        masked_frame[mask == True] = [255, 255, 255]
        masked_frame[mask == False] = [0, 0, 0]

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        min_line_length = 200
        max_line_gap = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, min_line_length, max_line_gap)

        img = np.zeros_like(frame)

        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        imgs.append(img)

        if verbose:
            print(len(imgs))

    save_frames_as_video(output_video_path, imgs, fps)


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


# takes frame, cuts off 15% of pixels from all sides (left, right, top, bottom),
# finds largest connected component and removes from mask other components
# center_ratio_size is the share of the image which will be cropped before finding
# largest component, finds convex hull
# as the result of experiment initial video with table layout will be saved
def exp3(input_video_path: str, output_video_path, center_ratio_size: float = 0.7):
    capture = cv2.VideoCapture(input_video_path)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    imgs = []

    while capture.isOpened():
        response, frame = capture.read()

        if not response:
            break

        print('!!')

        frame = frame[:, :, ::-1]

        mask = blue_table_mask(frame).astype('int')
        mask = find_largest_area_component_mask(mask, center_ratio_size)

        masked_frame = deepcopy(frame)
        masked_frame[mask == True] = [255, 255, 255]
        masked_frame[mask == False] = [0, 0, 0]

        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        min_line_length = 200
        max_line_gap = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, min_line_length, max_line_gap)

        points = np.array([(0, 0) for _ in range(2 * len(lines[:, 0, :]))])

        for i, (x1, y1, x2, y2) in enumerate(lines[:, 0, :]):
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            points[2 * i] = (x1, y1)
            points[2 * i + 1] = (x2, y2)

        # print(f'{len(points)} points before convex hull. ', end='')
        convex_hull = ConvexHull(points)
        hull_size = len(convex_hull.vertices)
        # print(f'{len(convex_hull.vertices)} points after convex hull')

        img = frame[:, :, ::-1]
        for i in range(hull_size):
            x1, y1 = points[convex_hull.vertices[i]]
            x2, y2 = points[convex_hull.vertices[(i + 1) % hull_size]]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 4)

        # plt.imshow(img[:, :, ::-1])
        # plt.show()
        # break

        imgs.append(img)

    save_frames_as_video(output_video_path, imgs, fps)


if __name__ == '__main__':
    # dir='007'
    # for i in range(109):
    #     exp1(f'resources/{dir}/raw/video{dir}_{i}.mp4', f'resources/{dir}/exp1/video{dir}_{i}_exp.mp4', verbose=False)
    #     exp2(f'resources/{dir}/raw/video{dir}_{i}.mp4', f'resources/{dir}/exp2/video{dir}_{i}_exp.mp4', verbose=False)
    # exp3('resources/001/raw/video007_22.mp4', 'hah.mp4', 0.5)
    exp3('resources/001/raw/video001_12.mp4', 'resources/001/exp3/video001_12_exp3.mp4', 0.5)
