import argparse
import typing
import cv2
import numpy as np

from video_operations import save_frames_as_video


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video splitter by layout'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Path to the video in mp4 format',
        required=True
    )
    parser.add_argument(
        '--layout',
        type=str,
        help='Path to the layout',
        required=True
    )
    return parser.parse_args()


# read layout file and parse
# returns the list of segments (tuples of int) in format
# [(l1, r1), (l2, r2), ..., (ln, rn)]
# where i-th shot is in segment [l_i, r_i]
# l_i, r_i in seconds
def read_shot_segments(layout_path: str) -> typing.List[typing.Tuple[int, int]]:
    def parse_seconds(t: str) -> int:
        secs = 0
        for part in list(map(int, t.split(':'))):
            secs = secs * 60 + part
        return secs
    segments = []
    with open(layout_path, 'r') as layout_file:
        for line in layout_file:
            l, r = line.split('-')
            l = parse_seconds(l)
            r = parse_seconds(r)
            segments.append((l, r))
    return segments


# splits video into peaces and saves each peace into separate files
def split_video(video_path: str, segments: typing.List[typing.Tuple[int, int]]):
    capture = cv2.VideoCapture(video_path)
    fps = np.round(capture.get(cv2.CAP_PROP_FPS))

    frame_id = 0
    segment_id = 0
    current_frames = []
    peace_id = 0

    video_path_prefix = video_path[:-4]  # removes .mp4 from path

    while capture.isOpened():
        while segment_id < len(segments) and segments[segment_id][1] < frame_id // fps + 1:
            if len(current_frames) > 0:
                peace_path = f'{video_path_prefix}_{peace_id}.mp4'
                save_frames_as_video(peace_path, current_frames, fps)
                peace_id += 1
                current_frames = []
            segment_id += 1
        if segment_id == len(segments):
            break
        response, frame = capture.read()
        if segments[segment_id][0] <= frame_id // fps + 1:
            current_frames.append(frame)
        frame_id += 1

    if len(current_frames) > 0:
        peace_path = f'{video_path_prefix}_{peace_id}.mp4'
        save_frames_as_video(peace_path, current_frames, fps)


def validate_segments(segments) -> bool:
    for l, r in segments:
        if l > r:
            return False
    for i in range(len(segments) - 1):
        l1, r1 = segments[i]
        l2, r2 = segments[i + 1]
        if r1 > l2:
            return False
    return True


if __name__ == '__main__':
    args = parse_args()

    video_path = args.video
    layout_path = args.layout

    assert video_path.endswith('.mp4')

    segments = read_shot_segments(layout_path)
    assert validate_segments(segments)

    split_video(video_path, segments)
