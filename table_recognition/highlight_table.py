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


if __name__ == '__main__':
    args = parse_args()

    capture = cv2.VideoCapture(args.video)
    fps = int(np.round(capture.get(cv2.CAP_PROP_FPS)))

    h = None
    w = None
    writer = None

    with open(args.layout) as layout_file:
        for line in layout_file:
            coordinates = list(map(int, line.strip().split(' ')))
            hull = [(coordinates[i + 1], coordinates[i]) for i in range(0, len(coordinates), 2)]

            response, frame = capture.read()
            assert response

            if h is None:
                h, w = frame.shape[: 2]
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

            hull_size = len(hull)

            for i in range(hull_size):
                x1, y1 = hull[i]
                x2, y2 = hull[(i + 1) % hull_size]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.circle(frame, (x1, y1), 10, (0, 0, 255), 4)

            writer.write(frame)

    writer.release()
