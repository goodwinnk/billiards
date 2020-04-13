from argparse import ArgumentParser
from pathlib import Path
import json

import cv2

from ball_detection.detection import get_hough_circles, get_circle_region_borders


def extract_candidates(arguments):
    data_dir = arguments.data_dir

    table_mask = cv2.imread(str(data_dir / arguments.table_mask_path))[:, :, 0]
    image_candidates = {}

    for image_path in data_dir.glob(arguments.picture_path_template):
        image = cv2.imread(str(image_path))

        circles = get_hough_circles(image, table_mask)
        candidate_regions = get_circle_region_borders(image, circles)
        image_candidates[image_path.name] = [
            {
                'center': [float(cx), float(cy)],
                'radius': float(r),
                'box': [x0, x1, y0, y1]
            }
            for (cx, cy), r, (x0, x1, y0, y1) in candidate_regions
        ]

    out_path = data_dir / arguments.output_path
    with out_path.open('wt') as out_file:
        json.dump(image_candidates, out_file, indent='  ')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    subparsers = arg_parser.add_subparsers()

    candidates_parser = subparsers.add_parser('extract_candidates')
    candidates_parser.add_argument('data_dir', type=Path)
    candidates_parser.add_argument('picture_path_template')
    candidates_parser.add_argument('-o', '--output_path', default='candidates.json')
    candidates_parser.add_argument('-m', '--table_mask_path', default='table_mask.png')
    candidates_parser.set_defaults(func=extract_candidates)

    args = arg_parser.parse_args()
    args.func(args)
