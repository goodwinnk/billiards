import json

import numpy as np
import cv2


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
        for region in image_regions:
            x0, x1, y0, y1 = region['box']
            box = image[y0:y1, x0:x1]
            box = cv2.resize(box, (32, 32))
            box = box / 255
            # box = np.float32(box > 0)
            # box = np.expand_dims(box, 2)
            boxes.append(box)
            ys.append(region['region_type'] != 'false_detection')

    boxes = np.array(boxes)
    ys = np.array(ys)
    return boxes, ys
