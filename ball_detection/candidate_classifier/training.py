import torch


def train(n_epochs, model, x_train, y_train, x_val, y_val, lr=1e-2, weight_decay=0., class_weights=None,
          save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    best_acc = 0
    for epoch in range(n_epochs):
        prediction = model(x_train)
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
