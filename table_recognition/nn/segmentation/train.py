import os
from typing import Tuple, List, Dict
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from table_recognition.nn.table_recognition_ds import TableRecognitionDataset
import matplotlib.pyplot as plt
from table_recognition.highlight_table import highlight_table_on_frame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import collections
import segmentation_models_pytorch as smp
from catalyst import utils
from skimage.io import imread as gif_imread
from torch import nn, optim
from catalyst.contrib.nn import DiceLoss, IoULoss, RAdam, Lookahead
from catalyst.dl import SupervisedRunner, DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from table_recognition.nn.augm import *
from catalyst.dl.utils import trace
import torch
from table_recognition.table_recognition_models import (
    TableRecognizer,
    NeuralNetworkBasedTableRecognizer,
    HeuristicsBasedTableRecognizer,
)
from table_recognition.find_table_polygon import find_table_polygon
from time import time
from albumentations.pytorch import ToTensor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script trains model on given dataset for table recognition task')
    parser.add_argument('--dataset', required=True, help='Path to the directory with dataset')
    parser.add_argument('--seed', required=False, type=str, default=42, help='Seed for random generators')
    return parser.parse_args()


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


def train_segmentation_model(
        model: torch.nn.Module,
        logdir: str,
        num_epochs: int,
        loaders: Dict[str, DataLoader]
):
    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss()
    }

    learning_rate = 0.001
    encoder_learning_rate = 0.0005

    layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}
    model_params = utils.process_model_params(model, layerwise_params=layerwise_params)
    base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
    optimizer = Lookahead(base_optimizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    device = utils.get_device()
    runner = SupervisedRunner(device=device, input_key='image', input_target_key='mask')

    callbacks = [
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),

        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",
            metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
        ),

        # metrics
        DiceCallback(input_key='mask'),
        IouCallback(input_key='mask'),
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=logdir,
        num_epochs=num_epochs,
        main_metric="iou",
        minimize_metric=False,
        verbose=True,
        load_best_on_end=True,
    )
    best_model_save_dir = os.path.join(logdir, 'save')
    os.makedirs(best_model_save_dir)
    torch.save(model, os.path.join(best_model_save_dir, 'best_model.pth'))   # save best model (by valid loss)
    batch = next(iter(loaders["valid"]))
    try:
        runner.trace(model=model, batch=batch, logdir=logdir, fp16=False)  # optimized version (not all models can be traced)
    except Exception:
        pass


class Statistics:

    def __init__(
            self,
            mean_iou: float,
            mean_dice: float,
            mean_time: float
    ):
        self.mean_iou = mean_iou
        self.mean_dice = mean_dice
        self.mean_time = mean_time


def get_statistics(loader: DataLoader, table_recognizer: TableRecognizer, verbose=False) -> Statistics:
    truth_ptr = iter(loader)
    images_ptr = iter(loader)

    sum_iou = 0
    sum_dice = 0
    cnt = 0
    sum_time = 0

    while True:
        st = time()
        masks = table_recognizer.next_mask()
        if masks is None:
            break
        sum_time += time() - st
        truths = next(truth_ptr)['mask']
        for mask, truth in zip(masks, truths):
            mask = mask.astype(bool)
            truth = truth.numpy().astype(bool)

            iou = np.sum(mask & truth) / np.sum(mask | truth)
            dice = 2 * np.sum(mask & truth) / (np.sum(mask) + np.sum(truth))

            sum_iou += iou
            sum_dice += dice
            cnt += 1

        if verbose:
            images = next(images_ptr)['image']
            for mask, truth, image in zip(masks, truths, images):
                plt.figure(figsize=(14, 10))
                plt.subplot(1, 3, 1)
                plt.imshow(mask)
                plt.title(f"Mask")
                plt.subplot(1, 3, 2)
                plt.imshow(truth.reshape(truth.shape[-2:]))
                plt.title(f'Truth')
                plt.subplot(1, 3, 3)

                image = utils.tensor_to_ndimage(image)
                image = (image * 255 + 0.5).astype(int).clip(0, 255).astype('uint8')

                plt.imshow(image)
                plt.show()

    return Statistics(
        mean_iou=sum_iou / cnt,
        mean_dice=sum_dice / cnt,
        mean_time=sum_time / cnt
    )


if __name__ == '__main__':
    args = parse_arguments()
    SEED = args.seed
    ROOT = Path(args.dataset)

    img_paths, targets = retrieve_dataset(ROOT)

    train_transforms = compose([
        resize_transforms(),
        hard_transforms(),
        post_transforms()
    ])
    valid_transforms = compose([pre_transforms(), post_transforms()])
    loaders = get_loaders(
        img_paths=img_paths,
        targets=targets,
        random_state=SEED,
        batch_size=8,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
    )

    def train_model(encoder_name: str, decoder_name: str, model_class_name, num_epochs: int, loaders: Dict[str, DataLoader]):
        train_segmentation_model(
            model=model_class_name(encoder_name=encoder_name, classes=1),
            logdir=f'./table_recognition/nn/segmentation/logs/{encoder_name}_{decoder_name}',
            num_epochs=num_epochs,
            loaders=loaders
        )

    def infer_model(encoder_name: str, decoder_name: str, load_mode: str ='trace', verbose: bool = False):
        if load_mode == 'trace':
            model = trace.load_traced_model(
                model_path=f'./table_recognition/nn/segmentation/logs/{encoder_name}_{decoder_name}/trace/traced-forward.pth',
                device='cpu'
            )                                                                       
        elif load_mode == 'load':
            model = torch.load(f'./table_recognition/nn/segmentation/logs/{encoder_name}_{decoder_name}/save/best_model.pth')
            model.to('cpu')
        else:
            raise Exception(f'Unresolved load mode = {load_mode}')
        table_recognizer = NeuralNetworkBasedTableRecognizer(model=model, loader=loaders['valid'])
        # table_recognizer = HeuristicsBasedTableRecognizer(loader=loaders['valid'])
        
        statistics = get_statistics(loaders['valid'], table_recognizer, verbose=verbose)
        
        print(f'IoU = {statistics.mean_iou}')
        print(f'DICE = {statistics.mean_dice}')
        print(f'time = {statistics.mean_time}')
        print(f'fps = {1 / statistics.mean_time}')


    # train_segmentation_model(
    #     model=smp.FPN(encoder_name="resnet18", classes=1),
    #     logdir='./table_recognition/nn/segmentation/logs/resnet34_FPN',
    #     num_epochs=25,
    #     loaders=loaders
    # )

    # train_segmentation_model(
    #     model=smp.FPN(encoder_name="efficientnet-b0", classes=1),
    #     logdir='./table_recognition/nn/segmentation/logs/efficientnet-b0_FPN',
    #     num_epochs=1,
    #     loaders=loaders
    # )

    # train_segmentation_model(
    #     model=smp.FPN(encoder_name="mobilenet_v2", classes=1),
    #     logdir='./table_recognition/nn/segmentation/logs/mobilenet_v2_FPN',
    #     num_epochs=1,
    #     loaders=loaders
    # )    

    # train_segmentation_model(
    #     model=smp.Unet(encoder_name="mobilenet_v2", classes=1),
    #     logdir='./table_recognition/nn/segmentation/logs/mobilenet_v2_Unet',
    #     num_epochs=1,
    #     loaders=loaders
    # )

    # model = trace.load_traced_model(
    #     model_path='./table_recognition/nn/segmentation/logs/resnet18_FPN/trace/traced-forward.pth',
    #     device='cpu'
    # )

    # model = trace.load_traced_model(
    #     model_path='./table_recognition/nn/segmentation/logs/mobilenet_v2_FPN/trace/traced-forward.pth',
    #     device='cpu'
    # )

    # model = torch.load('./table_recognition/nn/segmentation/logs/efficientnet-b0_FPN/save/best_model.pth')
    # model.to('cpu')

#==============================TRAIN====================================================================================================

    # train_model(encoder_name='resnet18', decoder_name='Linknet', model_class_name=smp.Linknet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='resnet18', decoder_name='PSPNet', model_class_name=smp.PSPNet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='resnet18', decoder_name='Unet', model_class_name=smp.Unet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='resnet18', decoder_name='FPN', model_class_name=smp.FPN, num_epochs=25, loaders=loaders)
    #========================================

    # train_model(encoder_name='mobilenet_v2', decoder_name='Linknet', model_class_name=smp.Linknet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='mobilenet_v2', decoder_name='PSPNet', model_class_name=smp.PSPNet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='mobilenet_v2', decoder_name='Unet', model_class_name=smp.Unet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='mobilenet_v2', decoder_name='FPN', model_class_name=smp.FPN, num_epochs=25, loaders=loaders)
    #========================================

    # train_model(encoder_name='efficientnet-b0', decoder_name='Linknet', model_class_name=smp.Linknet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='efficientnet-b0', decoder_name='PSPNet', model_class_name=smp.PSPNet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='efficientnet-b0', decoder_name='Unet', model_class_name=smp.Unet, num_epochs=25, loaders=loaders)
    # train_model(encoder_name='efficientnet-b0', decoder_name='FPN', model_class_name=smp.FPN, num_epochs=25, loaders=loaders)

#==============================INFERENCE NN====================================================================================================

    # infer_model(encoder_name='resnet18', decoder_name='Linknet')
    # infer_model(encoder_name='resnet18', decoder_name='PSPNet')
    # infer_model(encoder_name='resnet18', decoder_name='Unet')
    # infer_model(encoder_name='resnet18', decoder_name='FPN')

    # infer_model(encoder_name='mobilenet_v2', decoder_name='Linknet')
    # infer_model(encoder_name='mobilenet_v2', decoder_name='PSPNet', verbose=True)
    # infer_model(encoder_name='mobilenet_v2', decoder_name='Unet')
    # infer_model(encoder_name='mobilenet_v2', decoder_name='FPN')

    # infer_model(encoder_name='efficientnet-b0', decoder_name='Linknet', load_mode='load')
    # infer_model(encoder_name='efficientnet-b0', decoder_name='PSPNet', load_mode='load')
    # infer_model(encoder_name='efficientnet-b0', decoder_name='Unet', load_mode='load')
    # infer_model(encoder_name='efficientnet-b0', decoder_name='FPN', load_mode='load')

#==============================INFERENCE Heuristics BASED====================================================================================================

    # table_recognizer = HeuristicsBasedTableRecognizer(loader=loaders['valid'])
    # statistics = get_statistics(loaders['valid'], table_recognizer, verbose=False)    
    # print(f'IoU = {statistics.mean_iou}')
    # print(f'DICE = {statistics.mean_dice}')
    # print(f'time = {statistics.mean_time}')
    # print(f'fps = {1 / statistics.mean_time}')