import torch


def pr(prediction, y):
    prediction = prediction > .5
    prediction = prediction.float()
    tp = torch.sum(prediction * y)
    return tp / prediction.sum(), tp / y.sum()


def train(n_epochs, model, x_train, y_train, x_val, y_val, lr=1e-2, weight_decay=0.):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    loss_function = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        prediction = model(x_train).reshape(-1)
        loss = loss_function(prediction, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.drop = False
        val_prediction = model(x_val).reshape(-1)
        model.drop = True
        val_loss = loss_function(val_prediction, y_val)
        val_p, val_r = pr(val_prediction, y_val)
        print(f'loss: {loss:.4f}/{val_loss:.4f}, precision: {val_p:.3f}, recall: {val_r:.3f}')

