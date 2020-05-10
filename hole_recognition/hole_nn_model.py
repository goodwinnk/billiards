import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20*20*3, 128)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.act2 = nn.Softmax(dim=1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.flatten(start_dim=1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return x


class HoleDetector:
    def __init__(self):
        self.model = Net()

    def load(self, path='weights.pt'):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def save(self, path='weights.pt'):
        torch.save(self.model.state_dict(), path)

    def train(self, train_img: np.array, train_res: np.array, test_img=None, test_res=None, epochs=400):
        self.model.train()
        train_img = torch.from_numpy(train_img)
        train_res = torch.from_numpy(train_res).long()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            losses = []
            optimizer.zero_grad()
            prediction = self.predict(train_img)
            loss = loss_function(prediction, train_res)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())
            if epoch % 10 == 0:
                print('Epoch: {}'.format(epoch))
                print(' Loss: {}'.format(np.mean(losses)))
                if test_img is not None and test_res is not None:
                    acc, _ = self.test(test_img=test_img, test_res=test_res)
                    acc1, _ = self.test(test_img=np.array(train_img), test_res=np.array(train_res))
                    print('\tAccuracy: {}\t{}'.format(acc, acc1))

    def predict(self, test_img: np.array):
        return self.model(test_img)

    def test(self, test_img, test_res):
        test_img = torch.from_numpy(test_img)
        test_res = torch.from_numpy(test_res)
        self.model.eval()
        prediction = self.predict(test_img)
        preds = []
        acc = 1
        for i in range(len(test_res)):
            pr = torch.argmax(prediction[i]).item()
            preds.append(pr)
            if test_res[i] == pr:
                acc += 1
        return acc / test_res.shape[0], np.array(preds)


def construct_dataset(alpha=0.9):
    data = []
    data_holes_dir = '../data/sync/holes_dataset/holes'
    data_not_holes_dir = '../data/sync/holes_dataset/not_holes'

    def process_file(name, img):
        ext = name.split('.')[-1]
        if ext == 'png':
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    for _, _, files in os.walk(data_holes_dir):
        for file in files:
            img = cv2.imread(data_holes_dir + '/' + file)
            img = process_file(file, img)
            data.append((img, 1))
    for _, _, files in os.walk(data_not_holes_dir):
        for file in files:
            img = cv2.imread(data_not_holes_dir + '/' + file)
            img = process_file(file, img)
            data.append((img, 0))

    data = np.array(data)
    np.random.shuffle(data)

    train_img = []
    train_res = []
    test_img = []
    test_res = []
    n = int(data.shape[0] * alpha)
    for i in range(n):
        img, res = data[i]
        train_img.append(img)
        train_res.append(res)
    for i in range(n, data.shape[0]):
        img, res = data[i]
        test_img.append(img)
        test_res.append(res)

    train_img = np.float32(train_img) / 255.0
    train_res = np.float32(train_res)
    test_img = np.float32(test_img) / 255.0
    test_res = np.float32(test_res)

    return train_img, train_res, test_img, test_res


def show_results(test_img: np.array, test_res: np.array, predictions: np.array):
    holes = []
    not_holes = []

    for i in range(len(test_img)):
        res = predictions[i]
        if res != test_res[i]:
            if test_res[i] == 1:
                holes.append(test_img[i])
            else:
                not_holes.append(test_img[i])

    def plot_images(comment, img):
        f = plt.figure(num=comment)
        for j in range(len(img)):
            f.add_subplot(j // 10 + 1, len(img), j % 10 + 1)
            plt.imshow(cv2.cvtColor(np.float32(img[j]), cv2.COLOR_RGB2BGR))
            plt.axis('off')

    plot_images('Holes recognized as not holes:', holes)
    plot_images('Not holes recognized as holes:', not_holes)

    plt.show()


def prepare_model(show_test_res=False):
    """
    Creates and trains a model for recognizing billiard table holes.
    If show_test_res is set, plots all wrongly recognized test dataset images and total loss value.
    """
    alpha = 0.9 if show_test_res else 1
    train_img, train_res, test_img, test_res = construct_dataset(alpha)
    if alpha == 1:
        test_img = None
        test_res = None
    model.train(train_img, train_res)
    if show_test_res:
        acc, predictions = model.test(test_img, test_res)
        print('Accuracy: {}'.format(acc))
        show_results(test_img, test_res, predictions)
    return model


if __name__ == '__main__':
    model = prepare_model(True)
    model.save()
