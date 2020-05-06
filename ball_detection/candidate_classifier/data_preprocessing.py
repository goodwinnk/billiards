import json
from pathlib import Path
from collections import defaultdict
import random
from typing import Iterable

import numpy as np
import cv2
from torch.utils.data import IterableDataset

from ball_detection.commons import BallType
from ball_detection.candidate_classifier.model import NET_INPUT_SIZE
from ball_detection.candidate_classifier.augmentations import AugmentationApplier
from ball_detection.commons import CANDIDATE_PADDING_COEFFICIENT


LABEL_DIRS = (
    ('not_balls', BallType.FALSE),
    ('solid_balls', BallType.INTEGRAL),
    ('striped_balls', BallType.STRIPED),
    ('white_balls', BallType.WHITE),
    ('black_balls', BallType.BLACK)
)

LABELS_BALL_TYPES = {
    'false_detection': BallType.FALSE,
    'striped': BallType.STRIPED,
    'integral': BallType.INTEGRAL,
    'white': BallType.WHITE,
    'black': BallType.BLACK
}

SHIFT_VARIANCE = 10


def cut_box(image, box_coordinates):
    x0, x1, y0, y1 = box_coordinates
    box = image[y0:y1, x0:x1]
    box = cv2.resize(box, NET_INPUT_SIZE)
    box = np.float32(box) / 255
    return box


def cut_boxes(image: np.array, regions: Iterable):
    return np.array([cut_box(image, region) for region in regions])


def read_json_dataset_index(data_dir: Path, markup_filename: str):
    markup_path = data_dir / markup_filename
    with markup_path.open() as markup_file:
        markup = json.load(markup_file)

    dataset_index = {}
    for image_filename, regions in markup.items():
        if not regions:
            continue
        region_descs = [(region['box'], LABELS_BALL_TYPES[region['region_type']]) for region in regions]
        dataset_index[str(data_dir / image_filename)] = region_descs
    return dataset_index


def _read_dir_coordinates(dir_path: Path):
    image_candidates = defaultdict(list)
    for coordinates_file_path in dir_path.glob('*.txt'):
        _, image_id = coordinates_file_path.stem.split('_')
        cx, cy, r = map(int, coordinates_file_path.read_text().split())
        image_candidates[image_id].append((cx, cy, r))
    return image_candidates


def read_folder_dataset_index(data_dir: Path = Path('data/sync/dataset_solid_striped_sep'),
                              images_dir: Path = Path('data/sync/images_for_dataset')):
    index = defaultdict(list)
    for subdir_name, label in LABEL_DIRS:
        subdir_dict = _read_dir_coordinates(data_dir / subdir_name)
        for image_name, regions in subdir_dict.items():
            image_path = (images_dir / image_name).with_suffix('.png')
            image = cv2.imread(str(image_path))
            region_descs = []
            m, n = image.shape[:2]
            for (cx, cy, r) in regions:
                half_side = min(int(r * CANDIDATE_PADDING_COEFFICIENT), cx, cy, n - cx, m - cy)
                box = (int(cx - half_side), int(cx + half_side), int(cy - half_side), int(cy + half_side))
                region_descs.append((box, label))
            index[str(image_path)].extend(region_descs)
    return index


def merge_dataset_indexes(indexes: Iterable[dict]):
    common_index = defaultdict(list)
    for index in indexes:
        for image_path, regions in index.items():
            common_index[image_path].extend(regions)
    return common_index


def split_balls_false_detections(dataset_index: dict):
    balls_index, false_index = defaultdict(list), defaultdict(list)
    for image_path, regions in dataset_index.items():
        for box, label in regions:
            if label == BallType.FALSE:
                false_index[image_path].append((box, label))
            else:
                balls_index[image_path].append((box, label))
    return balls_index, false_index


class CandidatesDataset(IterableDataset):
    def __init__(self, index: dict, move_prob: float = 0., shuffle=False):
        super(CandidatesDataset, self).__init__()
        self.data = []
        for image_path, regions in index.items():
            image = cv2.imread(image_path)
            boxes, labels = zip(*regions)
            box_cuts = cut_boxes(image, boxes)
            image_cut_candidates = list(zip(boxes, labels, box_cuts))
            self.data.append((image_path, image_cut_candidates))
        self.move_prob = move_prob
        self.n = sum(map(len, index.values()))
        self.shuffle = shuffle

    def __iter__(self):
        data = random.sample(self.data, len(self.data)) if self.shuffle else self.data
        for image_path, image_candidates in data:
            if self.move_prob and np.random.binomial(1, self.move_prob):
                image = cv2.imread(image_path)
                m, n = image.shape[:2]
                for (x0, x1, y0, y1), label, _ in image_candidates:
                    shift_x, shift_y = np.random.normal(0, SHIFT_VARIANCE, 2).astype(np.int32)
                    shift_x = min(max(shift_x, -x0), n - x1)
                    shift_y = min(max(shift_y, -y0), m - y1)
                    shifted_box = x0 + shift_x, x1 + shift_x, y0 + shift_y, y1 + shift_y
                    yield cut_box(image, shifted_box), label
            else:
                for _, label, box_cut in image_candidates:
                    yield box_cut, label

    def __len__(self):
        return self.n


class MixDataset(IterableDataset):
    def __init__(self, source1: IterableDataset, source2: IterableDataset):
        super(MixDataset, self).__init__()
        self.len1 = len(source1)
        self.len2 = len(source2)
        self.source1 = source1
        self.source2 = source2

    def __iter__(self):
        pos1, pos2 = 0, 0
        iter1, iter2 = iter(self.source1), iter(self.source2)
        while True:
            if pos1 / self.len1 <= pos2 / self.len2:
                pos1 += 1
                yield next(iter1)
            else:
                pos2 += 1
                yield next(iter2)
            if pos1 == self.len1 and pos2 == self.len2:
                break

    def __len__(self):
        return self.len1 + self.len2


class LabeledImageDataset(IterableDataset):
    def __init__(self, source: IterableDataset, augmentation_applier: AugmentationApplier = None):
        super(LabeledImageDataset, self).__init__()
        self.source = source
        self.augmentation_applier = augmentation_applier

    def __iter__(self):
        for image, label in self.source:
            if self.augmentation_applier:
                image = self.augmentation_applier.apply(image)
            image = image.transpose((2, 0, 1))
            label = label.value
            yield image, label

    def __len__(self):
        return len(self.source)
