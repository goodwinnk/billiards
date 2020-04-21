from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np


def construct_dataset():
    data = []
    data_balls_dir = '../data/sync/dataset/balls'
    data_not_balls_dir = '../data/sync/dataset/not_balls'
    for _, _, files in os.walk(data_balls_dir):
        for file in files:
            img = cv2.imread(data_balls_dir + '/' + file)
            data.append((img, 1))
    for _, _, files in os.walk(data_not_balls_dir):
        for file in files:
            img = cv2.imread(data_not_balls_dir + '/' + file)
            data.append((img, 0))

    data = np.array(data)
    np.random.shuffle(data)

    train_img = []
    train_res = []
    test_img = []
    test_res = []
    N = 900
    for i in range(N):
        img, res = data[i]
        train_img.append(img)
        train_res.append(res)
    for i in range(N, data.shape[0]):
        img, res = data[i]
        test_img.append(img)
        test_res.append(res)

    train_img = np.float32(train_img) / 255.0
    train_res = np.float32(train_res)
    test_img = np.float32(test_img) / 255.0
    test_res = np.float32(test_res)

    return train_img, train_res, test_img, test_res


def create():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(20, 20, 3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train(model: keras.models, train_img: np.array, train_res: np.array):
    model.fit(train_img, train_res, epochs=20)


def predict(model, test_img, test_res):
    model.evaluate(test_img, test_res, verbose=2)
    predictions = model.predict(test_img)
    return predictions


def show_results(test_img: np.array, test_res: np.array, predictions: np.array):
    balls = []
    not_balls = []

    for i in range(len(test_img)):
        res = np.argmax(predictions[i])
        if res != test_res[i]:
            if test_res[i] == 1:
                balls.append(test_img[i])
            else:
                not_balls.append(test_img[i])

    def plot_images(comment, img):
        f = plt.figure(num=comment)
        for i in range(len(img)):
            f.add_subplot(i // 10 + 1, len(img), i % 10 + 1)
            plt.imshow(img[i])
            plt.axis('off')

    plot_images('Balls recognized as not balls:', balls)
    plot_images('Not balls recognized as balls:', not_balls)

    plt.show()


if __name__ == '__main__':
    """
    Trains a model using data/sync/dataset balls and not balls images.
    Then calculates the test predictions, prints the accuracy and
    plots all wrong classified images.
    """
    train_img, train_res, test_img, test_res = construct_dataset()
    model = create()
    train(model, train_img, train_res)
    predictions = predict(model, test_img, test_res)
    show_results(test_img, test_res, predictions)
