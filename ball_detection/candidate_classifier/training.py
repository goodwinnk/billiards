import torch
from ball_detection.candidate_classifier.model import CLASSIFICATION_SCORE_THRESHOLD


def pr(prediction, y):
    prediction = prediction > CLASSIFICATION_SCORE_THRESHOLD
    prediction = prediction.float()
    tp = torch.sum(prediction * y)
    return tp / prediction.sum(), tp / y.sum()


def train(n_epochs, model, x_train, y_train, x_val, y_val, lr=1e-2, weight_decay=0., class_weights=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    for epoch in range(n_epochs):
        prediction = model(x_train)
        loss = loss_function(prediction, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.drop = False
        val_prediction = model(x_val)
        model.drop = True
        val_loss = loss_function(val_prediction, y_val)
        acc = (val_prediction.argmax(axis=1) == y_val).float().mean()
        print(f'loss: {loss:.4f}/{val_loss:.4f}, accuracy: {acc}')

