from abc import abstractmethod

import numpy as np
import torch
from catalyst import utils
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from typing import Union


from table_recognition.find_table_polygon import (
    find_convex_hull_mask,
    find_table_polygon,
    remove_big_angles_from_hull,
    take_longest_sides_from_hull
)
from copy import deepcopy


class TableRecognizer:

    def __init__(self, loader: DataLoader):
        """
        Tool which can make prediction where billiards table is located on the image

        Args:
            loader: data loader from which recognizer will take batches and make predictions for them
        """
        self.iter = iter(loader)

    def next_mask(self) -> Union[np.ndarray, None]:
        """
        Takes next batch from the loader and predicts mask for it
        """
        try:
            return self.predict_table_mask(next(self.iter)['image'])
        except StopIteration:
            return None

    def next_polygon(self) -> Union[np.ndarray, None]:
        """
        Takes next batch from the loader and predicts table polygon for it
        """
        return self.predict_table_polygon(next(iter(self.iter))['image'])

    def predict_table_mask(self, batch: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predicts masks of table for each frame in batch and returns corresponding masks

        Args:
             batch: batch of frames.

        Returns:
            mask: batch of masks for each frame in input batch
        """
        pass

    @abstractmethod
    def predict_table_polygon(self, batch: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predicts table vertices for each frame in batch and returns it

        Args:
            batch: batch of frames
        Returns:
            polygons: batch of table polygons. Each polygon is the convex polygon with 4 vertices and
                all vertices have 2 coordinates (x, y).
                Polygon has the following form: [x1, y1, x2, y2, x3, y3, x4, y4] where x coordinates are responsible
                for polygon place relative to image width, y -- to image height
        """
        pass


class HeuristicsBasedTableRecognizer(TableRecognizer):
    """
        Table recognizer based on article: https://www.researchgate.net/profile/Anil_Kokaram/publication/221368680_Content_Based_Analysis_for_Video_from_Snooker_Broadcasts/links/54ceead30cf24601c09270cc/Content-Based-Analysis-for-Video-from-Snooker-Broadcasts.pdf
    """

    def predict_table_mask(self, batch: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        polygons = self.predict_table_polygon(batch)
        image_size = batch.shape[2: 4]
        return np.array([
            find_convex_hull_mask(image_size, list(zip(polygon[0::2], polygon[1::2])))
            for polygon in polygons
        ])

    def predict_table_polygon(self, batch: torch.Tensor) -> np.ndarray:
        hulls = []
        import matplotlib.pyplot as plt
        for frame in batch:
            image = utils.tensor_to_ndimage(frame)
            image = (image * 255 + 0.5).astype(int).clip(0, 255).astype('uint8')
            try:
                hull = find_table_polygon(deepcopy(image))
                hull = remove_big_angles_from_hull(hull)
                hull = take_longest_sides_from_hull(hull, 4)
                hulls.append(hull.reshape(-1))
            except Exception:
                hulls.append(np.array([0, 0, 0, 1, 1, 1, 1, 0]))
        return np.array(hulls)


class NeuralNetworkBasedTableRecognizer(TableRecognizer):
    """
        Table recognizer which uses neural network inside

        Arguments:
            model: neural network model
    """
    def __init__(self, model: torch.nn.Module, loader: DataLoader):
        super().__init__(loader)
        self.model = model

    def predict_table_mask(self, batch: torch.Tensor) -> np.ndarray:
        self.model.eval()
        return np.array([
            utils.detach(logits[0].sigmoid() > 0.5).astype(bool)
            for logits in self.model(batch)
        ])
