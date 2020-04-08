from hmmlearn import hmm
import numpy as np
import cv2
import matplotlib.pyplot as plt

from copy import deepcopy

from src.video_operations import save_frames_as_video

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


if __name__ == '__main__':
    dir='007'
    for i in range(24):
        exp1(f'resources/{dir}/video{dir}_{i}.mp4', f'resources/{dir}/exp1/video{dir}_{i}_exp.mp4', verbose=False)
        exp2(f'resources/{dir}/video{dir}_{i}.mp4', f'resources/{dir}/exp2/video{dir}_{i}_exp.mp4', verbose=False)
