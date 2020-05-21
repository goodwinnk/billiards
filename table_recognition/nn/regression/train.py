import argparse
import os
from pathlib import Path

import catalyst.dl as dl
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from catalyst import utils
from catalyst.contrib.nn import DiceLoss, IoULoss, RAdam, Lookahead
from torch import optim

from table_recognition.find_table_polygon import find_convex_hull_mask, get_canonical_4_polygon
from table_recognition.nn.augm import compose, resize_transforms, hard_transforms, post_transforms, pre_transforms
from table_recognition.nn.table_recognition_ds import retrieve_dataset, get_loaders
from table_recognition.table_recognition_models import (
    NNRegressionBasedTableRecognizer,
    get_statistics
)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script trains model on given dataset for table recognition task')
    parser.add_argument('--dataset', required=True, help='Path to the directory with dataset')
    parser.add_argument('--seed', required=False, type=str, default=42, help='Seed for random generators')
    return parser.parse_args()


def MAELoss(v1, v2):
    return torch.mean((v1[:, 0] - v2[:, 0]) ** 2)


def smooth_l1(v1, v2):
    res = abs(v1 - v2)
    lt1 = res < 1
    ge1 = res >= 1
    res[lt1] = 0.5 * (res[lt1] ** 2)
    res[ge1] = res[ge1] - 0.5
    return torch.mean(res)


class Net(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 64)
        self.fc = nn.Linear(64, 8)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class CustomRunner(dl.Runner):

    def _handle_batch(self, batch):
        x = batch['image']
        y = batch['table']
        y_pred = self.model(x)
        # loss = smooth_l1(y_pred, y)
        # loss = nn.MSELoss()(y_pred, y)
        loss = 0.7 * nn.MSELoss()(y_pred, y) + 0.3 * MAELoss(y_pred, y)

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()

        img_size = x[0].shape[1:]
        iou = 0
        dice = 0
        cnt = 0
        for true_table, pred_table in zip(y, y_pred):
            true_table = get_canonical_4_polygon(true_table.cpu().detach().numpy())
            pred_table = get_canonical_4_polygon(pred_table.cpu().detach().numpy())
            # print(true_table)
            # print(pred_table)
            true_mask = find_convex_hull_mask(
                img_size,
                [(int(x * img_size[0]),
                  int(y * img_size[1]))
                 for x, y in true_table]
            ).astype(bool)
            pred_mask = find_convex_hull_mask(
                img_size,
                [(int(x * img_size[0]),
                  int(y * img_size[1]))
                 for x, y in pred_table]
            ).astype(bool)
            I = np.sum(true_mask & pred_mask)
            U = np.sum(true_mask | pred_mask)
            iou += I / U
            dice += 2 * I / (np.sum(true_mask) + np.sum(pred_mask))
            cnt += 1

        self.state.batch_metrics.update({
            'loss': loss,
            'iou': iou / cnt,
            'dice': dice / cnt}
        )


class RegressionFromSegmentation(nn.Module):

    def __init__(self, segmentation_model: nn.Module):
        super().__init__()
        self.encoder = segmentation_model.encoder
        self.decoder = segmentation_model.decoder
        self.segmentation_head = segmentation_model.segmentation_head

        self.regression_head = nn.Sequential(
            nn.Conv2d(512, 1, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(784, 8)

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.decoder(x)

        # print(x.shape)

        y = self.regression_head(x)
        y = y.view(y.shape[0], -1)
        y = self.fc(y)
        # print(f'y_shape = {y.shape}')

        x = self.segmentation_head(x)

        # print(x.shape)

        return x, y


class CustomRunner2(dl.Runner):

    def _handle_batch(self, batch):
        x = batch['image']
        y = batch['table']
        y_pred_mask, y_pred_table = self.model(x)
        target_mask = batch['mask']
        # loss = smooth_l1(y_pred, y)
        # loss = nn.MSELoss()(y_pred, y)
        loss_reg = 0.7 * nn.MSELoss()(y_pred_table, y) + 0.3 * MAELoss(y_pred_table, y)

        segm_iou = IoULoss()(y_pred_mask, target_mask)
        segm_dice = DiceLoss()(y_pred_mask, target_mask)
        loss_segm_bce = nn.BCEWithLogitsLoss()(y_pred_mask, target_mask)

        loss_segm_iou = -segm_iou + 1
        loss_segm_dice = -segm_dice + 1

        loss_segm = loss_segm_iou + loss_segm_dice + 0.8 * loss_segm_bce

        loss = 10 * loss_reg + 0.1 * loss_segm

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()

        img_size = x[0].shape[1:]
        iou = 0
        dice = 0
        cnt = 0
        for true_table, pred_table in zip(y, y_pred_table):
            true_table = get_canonical_4_polygon(true_table.cpu().detach().numpy())
            pred_table = get_canonical_4_polygon(pred_table.cpu().detach().numpy())
            # print(true_table)
            # print(pred_table)
            true_mask = find_convex_hull_mask(
                img_size,
                [(int(x * img_size[0]),
                  int(y * img_size[1]))
                 for x, y in true_table]
            ).astype(bool)
            pred_mask = find_convex_hull_mask(
                img_size,
                [(int(x * img_size[0]),
                  int(y * img_size[1]))
                 for x, y in pred_table]
            ).astype(bool)
            I = np.sum(true_mask & pred_mask)
            U = np.sum(true_mask | pred_mask)
            iou += I / U
            dice += 2 * I / (np.sum(true_mask) + np.sum(pred_mask))
            cnt += 1

        self.state.batch_metrics.update({
            'loss': loss,
            'iou': iou / cnt,
            'dice': dice / cnt,

            'segm_iou': segm_iou,
            'segm_dice': segm_dice,

            'loss_segm': loss_segm,
            'loss_reg': loss_reg
        })


def simple_way():
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

    logdir = './table_recognition/nn/regression/logs5/'

    # model = models.resnet18(pretrained=True)
    # model.fc = nn.Linear(model.fc.in_features, 8)
    model = Net(models.resnet18(pretrained=True))

    # model = torch.load(f'{logdir}/save/best_model.pth')
    # model.to('cpu')

    # for batch in loaders['valid']:
    #     tables = model(batch['image'])
    #     for image, table in zip(batch['image'], tables):
    #         image = utils.tensor_to_ndimage(image)
    #         image = (image * 255 + 0.5).astype(int).clip(0, 255).astype('uint8')
    #         img_size = image.shape[:2]
    #         table = get_canonical_4_polygon(table)
    #         mask = find_convex_hull_mask(
    #             img_size,
    #             [(int(x * img_size[0]),
    #               int(y * img_size[1]))
    #              for x, y in table]
    #         ).astype(bool)
    #
    #         plt.figure(figsize=(14, 10))
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(image)
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(mask)
    #         plt.show()

    learning_rate = 0.001
    encoder_learning_rate = 0.0005

    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate, weight_decay=0.00003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    device = utils.get_device()
    # runner = SupervisedRunner(device=device, input_key='image', input_target_key='table')
    runner = CustomRunner(device=device)

    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=50,
        verbose=True,
        load_best_on_end=True,
        main_metric='loss'
    )

    best_model_save_dir = os.path.join(logdir, 'save')
    os.makedirs(best_model_save_dir, exist_ok=True)
    torch.save(model, os.path.join(best_model_save_dir, 'best_model.pth'))  # save best model (by valid loss)
    batch = next(iter(loaders["valid"]))
    try:
        runner.trace(model=model, batch=batch, logdir=logdir,
                     fp16=False)  # optimized version (not all models can be traced)
    except Exception:
        pass


def smart_way():
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

    logdir = './table_recognition/nn/regression/logs6/'

    model = torch.load(f'./table_recognition/nn/segmentation/logs/resnet18_PSPNet/save/best_model.pth')
    model: RegressionFromSegmentation = RegressionFromSegmentation(model)
    model.to(utils.get_device())

    learning_rate = 0.001
    encoder_learning_rate = 0.0005

    layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}
    model_params = utils.process_model_params(model, layerwise_params=layerwise_params)
    base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
    optimizer = Lookahead(base_optimizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    device = utils.get_device()

    runner = CustomRunner2(device=device)
    runner.train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=logdir,
        num_epochs=1000,
        verbose=True,
        load_best_on_end=True,
        main_metric='loss'
    )

    best_model_save_dir = os.path.join(logdir, 'save')
    os.makedirs(best_model_save_dir, exist_ok=True)
    torch.save(model, os.path.join(best_model_save_dir, 'best_model.pth'))  # save best model (by valid loss)
    batch = next(iter(loaders["valid"]))
    try:
        runner.trace(model=model, batch=batch, logdir=logdir,
                     fp16=False)  # optimized version (not all models can be traced)
    except Exception:
        pass

    # print(model)
    #
    # for batch in loaders['valid']:
    #     x = batch['image']
    #     logits = model(x)
    #     # exit(0)
    #     for image, logit in zip(batch['image'], logits):
    #
    #         # print(logit)
    #         # print(type(logit))
    #         # print(logit.dtype)
    #         # print(logit.shape)
    #         # print(logit.min(), logit.max(), logit.mean(), logit.std())
    #         # exit(0)
    #
    #         image = utils.tensor_to_ndimage(image)
    #         image = (image * 255 + 0.5).astype(int).clip(0, 255).astype('uint8')
    #         img_size = image.shape[:2]
    #
    #         mask = np.array(
    #             utils.detach(logit[0].sigmoid() > 0.5).astype(bool)
    #         )
    #
    #         plt.figure(figsize=(14, 10))
    #         plt.subplot(1, 2, 1)
    #         plt.imshow(image)
    #         plt.subplot(1, 2, 2)
    #         plt.imshow(mask)
    #         plt.show()


def show():
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
        batch_size=4,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
    )

    model = torch.load(f'./table_recognition/nn/regression/logs6/save/best_model.pth')
    # model = torch.load(f'./table_recognition/nn/segmentation/logs/resnet18_PSPNet/save/best_model.pth')
    model.to('cpu')
    model.eval()

    recognizer = NNRegressionBasedTableRecognizer(
        model=model,
        loader=loaders['valid']
    )

    # recognizer = NeuralNetworkBasedTableRecognizer(
    #     model=model,
    #     loader=loaders['valid']
    # )

    statistics = get_statistics(loaders['valid'], recognizer, verbose=False)

    print(statistics)


if __name__ == '__main__':
    show()
    # TODO: refactor dirty code
