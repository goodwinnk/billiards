import json

import numpy as np
import cv2

from ball_detection.candidate_classifier.model import NET_INPUT_SIZE


def cut_boxes(image, regions):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.Canny(image, 100, 200, 5)
    boxes = []
    for x0, x1, y0, y1 in regions:
        box = image[y0:y1, x0:x1]
        box = cv2.resize(box, NET_INPUT_SIZE)
        box = box / 255
        # box = np.float32(box > 0)
        # box = np.expand_dims(box, 2)
        boxes.append(box)
    boxes = np.array(boxes)
    return boxes


def read_data(data_dir):
    markup_path = data_dir / 'markup.json'
    with markup_path.open() as markup_file:
        markup = json.load(markup_file)

    boxes, ys = [], []

    for image_path, image_regions in markup.items():
        image = cv2.imread(str(data_dir / image_path))
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.Canny(image, 100, 200, 5)
        cur_image_boxes = cut_boxes(image, (region['box'] for region in image_regions))
        boxes.append(cur_image_boxes)
        ys.extend([region['region_type'] != 'false_detection' for region in image_regions])

    boxes = np.concatenate(boxes)
    ys = np.array(ys)
    return boxes, ys
