from abc import abstractmethod
from copy import deepcopy
from time import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from catalyst import utils
from torch.utils.data import DataLoader


import table_recognition.find_table_polygon as tp


class TableRecognizer:
    """
    Default implementation is the table recognizer
    based on article: https://www.researchgate.net/profile/Anil_Kokaram/publication/221368680_Content_Based_Analysis_for_Video_from_Snooker_Broadcasts/links/54ceead30cf24601c09270cc/Content-Based-Analysis-for-Video-from-Snooker-Broadcasts.pdf
    """

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

    def next_data(self):
        """
        Finds next table mask and table vertices and returns it as a pair
        """
        try:
            batch = next(self.iter)['image']
            mask = self.predict_table_mask(batch)
            table = self.predict_table_polygon(batch)
            return mask, table
        except StopIteration:
            return None, None

    def next_polygon(self) -> Union[np.ndarray, None]:
        """
        Takes next batch from the loader and predicts table polygon for it
        """
        try:
            return self.predict_table_polygon(next(iter(self.iter))['image'])
        except StopIteration:
            return None

    def predict_table_mask(self, batch: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predicts masks of table for each frame in batch and returns corresponding masks

        Args:
             batch: batch of frames.

        Returns:
            mask: batch of masks for each frame in input batch
        """
        polygons = self.predict_table_polygon(batch)
        image_size = batch.shape[2: 4]
        return np.array([
            tp.find_convex_hull_mask(
                image_size,
                [(int(x * image_size[0]), int(y * image_size[1]))
                 for x, y in tp.get_canonical_4_polygon(polygon)]
            )
            for polygon in polygons
        ])

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
        hulls = []
        for frame in batch:
            image = utils.tensor_to_ndimage(frame)
            image = (image * 255 + 0.5).astype(int).clip(0, 255).astype('uint8')
            try:
                hull = tp.find_table_polygon(deepcopy(image))
                hull = tp.remove_big_angles_from_hull(hull)
                hull = tp.take_longest_sides_from_hull(hull, 4).reshape(-1).astype('float32')
                hull[0::2] = hull[0::2] / image.shape[0]
                hull[1::2] = hull[1::2] / image.shape[1]
                hulls.append(hull)
            except Exception:
                hulls.append(np.array([0, 0, 0, 1, 1, 1, 1, 0]))
        return np.array(hulls)


class RecognitionStatistics:

    def __init__(
            self,
            mean_mask_iou: float,
            mean_mask_dice: float,
            mean_table_iou: float,
            mean_table_dice: float,
            mean_time: float
    ):
        self.mean_mask_iou = mean_mask_iou
        self.mean_mask_dice = mean_mask_dice
        self.mean_table_iou = mean_table_iou
        self.mean_table_dice = mean_table_dice
        self.mean_time = mean_time

    def __str__(self):
        return f'IoU_M = {self.mean_mask_iou}, Dice_M = {self.mean_mask_dice}, ' \
               f'IoU_T = {self.mean_table_iou}, Dice_T = {self.mean_table_dice}, ' \
               f'Time = {self.mean_time}, FPS = {1 / self.mean_time}'


def get_statistics(
        loader: DataLoader,
        table_recognizer: TableRecognizer,
        verbose=False
) -> RecognitionStatistics:

    ptr = iter(loader)

    sum_mask_iou = 0
    sum_mask_dice = 0
    sum_table_iou = 0
    sum_table_dice = 0
    cnt = 0
    sum_time = 0

    while True:
        st = time()
        masks, tables = table_recognizer.next_data()
        if masks is None:
            break
        sum_time += time() - st
        batch = next(ptr)
        img_size = batch['image'].shape[2:]
        for mask, truth, table, image in zip(masks, batch['mask'], tables, batch['image']):
            mask = mask.astype(bool)
            truth = truth.numpy().astype(bool)

            canonical_table = tp.get_canonical_4_polygon(table)
            table_mask = tp.find_convex_hull_mask(
                img_size,
                [(int(x * img_size[0]),
                  int(y * img_size[1]))
                 for x, y in canonical_table]
            ).astype(bool)

            mask_iou = np.sum(mask & truth) / np.sum(mask | truth)
            mask_dice = 2 * np.sum(mask & truth) / (np.sum(mask) + np.sum(truth))
            table_iou = np.sum(table_mask & truth) / np.sum(table_mask | truth)
            table_dice = 2 * np.sum(table_mask & truth) / (np.sum(table_mask) + np.sum(truth))

            sum_mask_iou += mask_iou
            sum_mask_dice += mask_dice
            sum_table_iou += table_iou
            sum_table_dice += table_dice
            cnt += 1

            if verbose:
                plt.figure(figsize=(14, 10))

                plt.subplot(2, 2, 1)
                plt.imshow(mask)
                plt.title('Predicted mask')

                plt.subplot(2, 2, 2)
                plt.imshow(truth.reshape(truth.shape[-2:]))
                plt.title('True mask')

                plt.subplot(2, 2, 3)
                image = utils.tensor_to_ndimage(image)
                image = (image * 255 + 0.5).astype(int).clip(0, 255).astype('uint8')
                plt.title('Image')
                plt.imshow(image)

                plt.subplot(2, 2, 4)
                plt.title('Predicted table mask')
                plt.imshow(table_mask)

                plt.show()

    return RecognitionStatistics(
        mean_mask_iou=sum_mask_iou / cnt,
        mean_mask_dice=sum_mask_dice / cnt,
        mean_table_iou=sum_table_iou / cnt,
        mean_table_dice=sum_table_dice / cnt,
        mean_time=sum_time / cnt
    )


class NNSegmentationBasedRecognizer(TableRecognizer):
    """
        Table recognizer which uses neural network inside

        Arguments:
            model: neural network model
    """
    def __init__(self, model: torch.nn.Module, loader: DataLoader):
        super().__init__(loader)
        self.model = model

    def predict_table_polygon(self, batch: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        hulls = []
        masks = self.predict_table_mask(batch)
        for frame, mask in zip(batch, masks):
            image = utils.tensor_to_ndimage(frame)
            image = (image * 255 + 0.5).astype(int).clip(0, 255).astype('uint8')
            try:
                hull = tp.find_table_polygon(deepcopy(image), mask=mask)
                # TODO: refactor (code duplication with super class default implementation)
                hull = tp.remove_big_angles_from_hull(hull)
                hull = tp.take_longest_sides_from_hull(hull, 4).reshape(-1).astype('float32')
                hull[0::2] = hull[0::2] / image.shape[0]
                hull[1::2] = hull[1::2] / image.shape[1]
                hulls.append(hull)
            except Exception:
                hulls.append(np.array([0, 0, 0, 1, 1, 1, 1, 0]))
        return np.array(hulls)

    def predict_table_mask(self, batch: torch.Tensor) -> np.ndarray:
        return np.array([
            utils.detach(logits[0].sigmoid() > 0.5).astype(bool)
            for logits in self.model(batch)
        ])


class NNRegressionBasedTableRecognizer(TableRecognizer):
    """
        Table recognizer which uses deep neural networks for table mask prediction
        and table vertices prediction simultaneously

        Arguments:
            model: neural network model which returns tuple of mask and table vertices
    """
    def __init__(self, model: torch.nn.Module, loader):
        super().__init__(loader)
        self.model = model

    def next_data(self):
        # Code copy pasted for performance reasons
        try:
            batch = next(self.iter)['image']
            logits, tables = self.model(batch)
            mask = np.array([
                utils.detach(logit[0].sigmoid() > 0.07).astype(bool)
                for logit in logits
            ])
            table = tables.detach().numpy()
            return mask, table
        except StopIteration:
            return None, None

    def predict_table_mask(self, batch: torch.Tensor) -> np.ndarray:
        return np.array([
            utils.detach(logit[0].sigmoid() > 0.07).astype(bool)
            for logit in self.model(batch)[0]
        ])

    def predict_table_polygon(self, batch: torch.Tensor):
        return self.model(batch)[1].detach().numpy()
