from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import collections
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

from table_recognition.find_table_polygon import find_convex_hull_mask


class TableRecognitionDataset(Dataset):

    def __init__(
            self,
            img_paths: np.ndarray,
            targets: np.ndarray,
            img_size: Tuple[int, int],  # (width, height),
            transforms=None
    ) -> None:
        self.img_paths = img_paths
        self.targets = targets
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.img_paths[idx]
        img = Image.open(img_path)

        table = np.copy(self.targets[idx, :8])
        # pocket = np.copy(self.targets[idx, 8])

        table[0::2] /= img.size[0]
        table[1::2] /= img.size[1]
        img = np.array(img.resize(self.img_size))

        table_mask = find_convex_hull_mask(
            img.shape[: 2],
            [(int(table[2 * i] * img.shape[0]),
              int(table[2 * i + 1] * img.shape[1]))
             for i in range(4)]
        )

        result = {
            'image': img,
            'mask': table_mask
        }

        if self.transforms is not None:
            result = self.transforms(**result)

        result['table'] = torch.Tensor(table.astype('float64')).float()

        return result


# Recursively collects all *.csv files, concat them and returns pair of:
# First is the numpy array of absolute paths to the images
# Second is the numpy array of target
# List and array correspond to each other
def retrieve_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    layout_files = list(sorted(root.rglob('*.csv')))

    # Concatenates layout path and image path
    def concat_paths(lfile_dir, x):
        x[0] = Path(lfile_dir) / x[0]
        return x

    total_df = np.vstack([
        pd.read_csv(lfile, header=None)
            .apply(lambda x: concat_paths(lfile.parent, x), axis=1)
            .values
        for lfile in layout_files
    ])

    return total_df[:, 0], total_df[:, 1:]


def get_loaders(
    img_paths: np.ndarray,
    targets: np.ndarray,
    random_state: int,
    valid_size_ratio: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (224, 224),
    train_transforms_fn=None,
    valid_transforms_fn=None
) -> collections.OrderedDict:
    indices = np.arange(len(img_paths))

    train_indices, valid_indices = train_test_split(
      indices, test_size=valid_size_ratio, random_state=random_state, shuffle=True
    )

    train_ds = TableRecognitionDataset(img_paths[train_indices], targets[train_indices], img_size, train_transforms_fn)
    valid_ds = TableRecognitionDataset(img_paths[valid_indices], targets[valid_indices], img_size, valid_transforms_fn)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        dataset=valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    loaders = collections.OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    return loaders
