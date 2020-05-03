import numpy as np
import torch
from torch.utils.data import DataLoader

from ball_detection.candidate_classifier.model import Net


def _accuracy(prediction, target):
    return (prediction.argmax(axis=1) == target).float().mean()


def train(n_epochs: int, model: Net, data_train: DataLoader, data_val: DataLoader, lr: float = 1e-2,
          weight_decay: float = 0., class_weights: torch.Tensor = None, save_path=None, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_acc = 0
    for epoch in range(n_epochs):
        losses_train = []
        for x_batch, y_batch in data_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses_train.append(loss.detach().cpu().numpy())

        predictions_val = []
        targets_val = []
        losses_val = []
        for x_batch, y_batch in data_val:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            predictions_val.append(prediction.detach())
            targets_val.append(y_batch)
            losses_val.append(loss.detach().cpu().numpy())
        predictions_val = torch.cat(predictions_val)
        targets_val = torch.cat(targets_val)
        accuracy = _accuracy(predictions_val, targets_val)
        print(f'loss: {np.mean(losses_train):.4f}/{np.mean(losses_val):.4f}, accuracy: {accuracy}')
        if accuracy > best_acc and save_path is not None:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
