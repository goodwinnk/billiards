import argparse
import cv2
from data_utils.video_operations import save_frames_as_video
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Takes the video and frame by frame table layout given in the file and draws the table on video'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True
    )
    parser.add_argument(
        '--layout',
        type=str,
        required=True
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True
    )
    return parser.parse_args()


def draw_polygon_on_frame(frame, polygon):
    hull_size = len(polygon)

    for i in range(hull_size):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % hull_size]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.circle(frame, (x1, y1), 10, (0, 0, 255), 4)


def highlight_table_video(video_path: str, layout_file_path: str, output_video_path: str):
    capture = cv2.VideoCapture(video_path)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))
    frames = []
    with open(layout_file_path) as layout_file:
        for line in layout_file:
            coordinates = list(map(int, line.strip().split(' ')))
            hull = [(coordinates[i + 1], coordinates[i]) for i in range(0, len(coordinates), 2)]

            response, frame = capture.read()
            assert response

            draw_polygon_on_frame(frame, hull)

            frames.append(frame)
    save_frames_as_video(output_video_path, frames, fps)


if __name__ == '__main__':
    args = parse_args()
    highlight_table_video(args.video, args.layout, args.output)
