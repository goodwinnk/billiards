from time import time
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ball_detection.candidate_generation_fcn.model import BallLocationFCN


PREDICTION_THRESHOLD = .5


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    prediction = prediction.squeeze() > PREDICTION_THRESHOLD
    target = target.bool().squeeze()
    tp = float(torch.sum(target & prediction).cpu())
    precision = tp / (float(prediction.sum().cpu()) + .1)
    recall = tp / float(target.sum().cpu() + .1)
    return precision, recall


def add_second_class(prediction: torch.Tensor) -> torch.Tensor:
    prediction = prediction.squeeze(dim=1).unsqueeze(3)
    prediction = torch.cat([1 - prediction, prediction], axis=3)
    return prediction.permute((0, 3, 1, 2))


def train(model: BallLocationFCN, epochs: int, optimizer: torch.optim.Optimizer, class_weights: Optional[torch.Tensor],
          data_train: DataLoader, data_val: DataLoader, device: str = 'cpu') -> None:
    loss_func = nn.NLLLoss(weight=class_weights)
    model = model.to(device)
    for epoch in range(epochs):
        start_time = time()
        train_losses = []
        for batch_x, batch_y in data_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_func(add_second_class(pred), batch_y.long())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().numpy())

        val_losses, precisions, recalls = [], [], []
        for batch_x, batch_y in data_val:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x).detach()
            loss = loss_func(add_second_class(pred), batch_y.long())
            val_losses.append(loss.cpu().numpy())
            precision, recall = metrics(pred, batch_y)
            precisions.append(precision)
            recalls.append(recall)
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_precision = np.mean(precisions)
        val_recall = np.mean(recalls)
        time_delta = time() - start_time
        print(f'{train_loss:.3f} {val_loss:.3f} {val_precision:.3f} {val_recall:.3f} {time_delta:.1f}')
