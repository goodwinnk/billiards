import numpy as np
import torch

from ball_detection.candidate_classifier.model import Net
from ball_detection.candidate_classifier.augmentations import AugmentationApplier


def train(n_epochs: int, model: Net, x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array,
          lr: float = 1e-2, weight_decay: float = 0., class_weights: torch.Tensor = None, save_path=None,
          augmentation_applier: AugmentationApplier = None, device: str = 'cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_acc = 0
    y_train, y_val = torch.LongTensor(y_train).to(device), torch.LongTensor(y_val).to(device)
    x_val = torch.FloatTensor(x_val).to(device)
    x_val = x_val.permute(0, 3, 1, 2)
    for epoch in range(n_epochs):
        x_train_final = x_train
        if augmentation_applier:
            x_train_final = augmentation_applier.apply_batch(x_train)
        x_train_final = torch.FloatTensor(x_train_final).to(device)
        x_train_final = x_train_final.permute(0, 3, 1, 2)

        prediction = model(x_train_final)
        loss = loss_function(prediction, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.drop = False
        val_prediction = model(x_val).detach()
        model.drop = True
        val_loss = loss_function(val_prediction, y_val)
        acc = (val_prediction.argmax(axis=1) == y_val).float().mean()
        print(f'loss: {loss:.4f}/{val_loss:.4f}, accuracy: {acc}')
        if acc > best_acc and save_path is not None:
            torch.save(model.state_dict(), save_path)
