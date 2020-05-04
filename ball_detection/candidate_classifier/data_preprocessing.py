import json
from pathlib import Path

import numpy as np
import cv2
from torch.utils.data import Dataset

from ball_detection.utils import BallType
from ball_detection.candidate_classifier.model import NET_INPUT_SIZE
from ball_detection.candidate_classifier.augmentations import AugmentationApplier


DIR_FALSE_POSITIVE = 'not_balls'
DIR_INTEGRAL_BALLS = 'solid_balls'
DIR_STRIPED_BALLS = 'striped_balls'
DIR_WHITE_BALLS = 'white_balls'
DIR_BLACK_BALLS = 'black_balls'


def cut_boxes(image, regions):
    boxes = []
    for x0, x1, y0, y1 in regions:
        box = image[y0:y1, x0:x1]
        box = cv2.resize(box, NET_INPUT_SIZE)
        box = np.float32(box) / 255
        boxes.append(box)
    boxes = np.array(boxes)
    return boxes


LABELS_BALL_TYPES = {
    'false_detection': BallType.FALSE,
    'striped': BallType.STRIPED,
    'integral': BallType.INTEGRAL,
    'white': BallType.WHITE,
    'black': BallType.BLACK
}


def read_data(data_dir, markup_filename='markup.json'):
    markup_path = data_dir / markup_filename
    with markup_path.open() as markup_file:
        markup = json.load(markup_file)

    boxes, labels = [], []
    for image_path, image_regions in markup.items():
        if not image_regions:
            continue
        image = cv2.imread(str(data_dir / image_path))
        cur_image_boxes = cut_boxes(image, (region['box'] for region in image_regions))
        boxes.append(cur_image_boxes)
        labels.extend([LABELS_BALL_TYPES[region['region_type']].value for region in image_regions])
    boxes = np.concatenate(boxes)
    labels = np.int64(labels)

    return boxes, labels


def read_dir(dir_path):
    images = [cv2.imread(str(file_dir)) for file_dir in dir_path.glob('*')]
    images = list(filter(lambda x: x is not None, images))
    return np.float32(images) / 255


def read_dataset_folder(data_dir=Path('data/sync/dataset_solid_striped_sep')):
    false_positives = read_dir(data_dir / DIR_FALSE_POSITIVE)
    solid_balls = read_dir(data_dir / DIR_INTEGRAL_BALLS)
    striped_balls = read_dir(data_dir / DIR_STRIPED_BALLS)
    white_balls = read_dir(data_dir / DIR_WHITE_BALLS)
    black_balls = read_dir(data_dir / DIR_BLACK_BALLS)

    pictures = np.concatenate((false_positives, solid_balls, striped_balls, white_balls, black_balls))
    labels = np.int64(
        [BallType.FALSE.value] * len(false_positives) +
        [BallType.INTEGRAL.value] * len(solid_balls) +
        [BallType.STRIPED.value] * len(striped_balls) +
        [BallType.WHITE.value] * len(white_balls) +
        [BallType.BLACK.value] * len(black_balls)
    )

    return pictures, labels


class CandidateDataset(Dataset):
    def __init__(self, candidates: np.array, labels: np.array, augmentation_applier: AugmentationApplier = None,
                 device='cpu'):
        self.candidates = candidates
        self.labels = labels
        self.augmentation_applier = augmentation_applier
        self.device = device

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, item):
        image = self.candidates[item]
        label = self.labels[item]
        if self.augmentation_applier:
            image = self.augmentation_applier.apply(image)
        image = image.transpose((2, 0, 1))
        return image, label
