from argparse import ArgumentParser
from pathlib import Path
import json

import cv2


def _make_key_dict(items_list):
    return {ord(i[0]): i for i in items_list}


REGION_TYPES = 'integral', 'white', 'black', 'striped', 'false_detection'
REGION_TYPE_DICT = _make_key_dict(REGION_TYPES)
BALL_COLOURS = 'yellow', 'blue', 'red', 'purple', 'orange', 'green', 'crimson'
BALL_COLOUR_DICT = _make_key_dict(BALL_COLOURS)


def mark_region(region, image):
    x0, x1, y0, y1 = region['box']
    region_cut = image[y0:y1, x0:x1]

    cv2.imshow('Markup', region_cut)

    region_type, ball_colour = None, None
    while not region_type:
        key = cv2.waitKey(10000)
        if key in REGION_TYPE_DICT:
            print(REGION_TYPE_DICT[key])
            region_type = REGION_TYPE_DICT[key]
    region['region_type'] = region_type

    if region_type in ('striped', 'integral'):
        while not ball_colour:
            key = cv2.waitKey(10000)
            if key in BALL_COLOUR_DICT:
                print(BALL_COLOUR_DICT[key])
                ball_colour = BALL_COLOUR_DICT[key]
        region['ball_colour'] = ball_colour
    cv2.destroyAllWindows()


def mark_regions(args):
    data_dir = args.data_dir
    candidates_path = data_dir / args.candidates_file
    with open(candidates_path) as candidates_file:
        candidates = json.load(candidates_file)
    markup_path = data_dir / args.markup_file
    if markup_path.is_file():
        with open(markup_path) as markup_file:
            markup = json.load(markup_file)
    else:
        markup = {}

    for image_name, image_regions in candidates.items():
        if image_name in markup:
            continue

        image = cv2.imread(str(data_dir / image_name))
        print(f'Marking {image_name}')

        for candidate_region in image_regions:
            mark_region(candidate_region, image)

        markup[image_name] = image_regions
        with markup_path.open('wt') as markup_file:
            json.dump(markup, markup_file)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('data_dir', type=Path, help='Directory with images and candidate files')
    arg_parser.add_argument('--candidates_file', default='candidates.json',
                            help='Name of the file with candidate regions')
    arg_parser.add_argument('--markup_file', default='markup.json',
                            help='Name of the file with markup')

    mark_regions(arg_parser.parse_args())
