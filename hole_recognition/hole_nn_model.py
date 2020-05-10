from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


class Model:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(20, 20, 3)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def get_loss(self, test_img: np.array, test_res: np.array):
        return self.model.evaluate(test_img, test_res, verbose=2)

    def train(self, train_img: np.array, train_res: np.array):
        self.model.fit(train_img, train_res, epochs=10)

    def predict(self, test_img: np.array):
        return self.model.predict(test_img)


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
        res = np.argmax(predictions[i])
        if res != test_res[i]:
            if test_res[i] == 1:
                holes.append(test_img[i])
            else:
                not_holes.append(test_img[i])

    def plot_images(comment, img):
        f = plt.figure(num=comment)
        for j in range(len(img)):
            f.add_subplot(j // 10 + 1, len(img), j % 10 + 1)
            plt.imshow(cv2.cvtColor(img[j], cv2.COLOR_RGB2BGR))
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
    model = Model()
    model.train(train_img, train_res)
    if show_test_res:
        model.get_loss(test_img, test_res)
        predictions = model.predict(test_img)
        show_results(test_img, test_res, predictions)
    return model


if __name__ == '__main__':
    prepare_model(True)
