from argparse import ArgumentParser
from pathlib import Path
import json

import cv2
import numpy as np

from ball_detection.candidate_generation_hough import HoughCircleDetector
from ball_detection.candidate_generation_motion import get_background, MotionDetector


def extract_candidates(data_dir, picture_name_template, output_filename, candidates_extractor):
    image_candidates = {}
    for image_path in data_dir.glob(picture_name_template):
        image = cv2.imread(str(image_path))
        image_candidates[image_path.name] = candidates_extractor(image)

    out_path = data_dir / output_filename
    with out_path.open('wt') as out_file:
        json.dump(image_candidates, out_file, indent='  ')


def extract_candidates_hough(arguments):
    data_dir = arguments.data_dir
    table_mask = cv2.imread(str(data_dir / arguments.table_mask_filename))[:, :, 0]
    circle_detector = HoughCircleDetector(table_mask)

    def get_circle_regions(image):
        return [
            {
                'center': [float(cx), float(cy)],
                'radius': float(r),
                'box': [x0, x1, y0, y1]
            }
            for (cx, cy), (x0, x1, y0, y1), r in circle_detector.get_regions(image, return_radius=True)
        ]

    extract_candidates(data_dir, arguments.picture_name_template, arguments.output_filename, get_circle_regions)


def extract_candidates_motion(arguments):
    data_dir = arguments.data_dir
    table_mask = cv2.imread(str(data_dir / arguments.table_mask_filename))[:, :, 0].astype(np.bool)
    background = cv2.imread(str(data_dir / arguments.background_filename))
    motion_detector = MotionDetector(table_mask, background)

    def get_motion_regions(image):
        return [
            {
                'center': list(center),
                'box': list(box)
            }
            for center, box in motion_detector.get_regions(image)
        ]

    extract_candidates(data_dir, arguments.picture_name_template, arguments.output_filename, get_motion_regions)


def extract_background(arguments):
    data_dir = arguments.data_dir
    table_mask = cv2.imread(str(data_dir / arguments.table_mask_path))[:, :, 0].astype(np.bool)
    video = cv2.VideoCapture(str(data_dir / arguments.video_name))

    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    frames_start = int(frame_rate * arguments.start)
    frames_step = int(frame_rate * arguments.step)
    frames_end = int(frame_rate * arguments.end)

    def read_frames():
        for frame_id in range(frames_start, frames_end, frames_step):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read()
            if ret:
                yield frame

    background = get_background(table_mask, read_frames())
    cv2.imwrite(str(data_dir / arguments.output_name), background)


def show_candidates(arguments):
    data_dir = arguments.data_dir
    with (data_dir / arguments.candidates_filename).open() as candidates_file:
        images_ball_candidates = json.load(candidates_file)

    for image_filename, candidates in images_ball_candidates.items():
        print(image_filename)
        image = cv2.imread(str(data_dir / image_filename))
        for candidate in candidates:
            y0, y1, x0, x1 = candidate['box']
            cv2.rectangle(image, (y0, x0), (y1, x1), (0, 255, 0), 2)

        cv2.imshow('Candidates', image)

        while True:
            key = cv2.waitKey(10000)
            if key == ord('n'):
                break
            elif key == ord('q'):
                return


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    subparsers = arg_parser.add_subparsers()

    hough_parser = subparsers.add_parser('candidates_hough')
    hough_parser.add_argument('data_dir', type=Path)
    hough_parser.add_argument('picture_name_template')
    hough_parser.add_argument('-o', '--output_filename', default='candidates.json')
    hough_parser.add_argument('-m', '--table_mask_filename', default='table_mask.png')
    hough_parser.set_defaults(func=extract_candidates_hough)

    motion_parser = subparsers.add_parser('candidates_motion')
    motion_parser.add_argument('data_dir', type=Path)
    motion_parser.add_argument('picture_name_template')
    motion_parser.add_argument('-b', '--background_filename', default='background.png')
    motion_parser.add_argument('-o', '--output_filename', default='candidates_motion.json')
    motion_parser.add_argument('-m', '--table_mask_filename', default='table_mask.png')
    motion_parser.set_defaults(func=extract_candidates_motion)

    background_parser = subparsers.add_parser('background')
    background_parser.add_argument('data_dir', type=Path)
    background_parser.add_argument('video_name')
    background_parser.add_argument('--start', type=float)
    background_parser.add_argument('--step', type=float)
    background_parser.add_argument('--end', type=float)
    background_parser.add_argument('-o', '--output_name', default='background.png')
    background_parser.add_argument('-m', '--table_mask_path', default='table_mask.png')
    background_parser.set_defaults(func=extract_background)

    show_parser = subparsers.add_parser('show')
    show_parser.add_argument('data_dir', type=Path)
    show_parser.add_argument('-c', '--candidates_filename', default='candidates.json')
    show_parser.set_defaults(func=show_candidates)

    args = arg_parser.parse_args()
    args.func(args)
