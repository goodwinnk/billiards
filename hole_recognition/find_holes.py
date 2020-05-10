import sympy.geometry as geom
import numpy as np
import cv2


def split_border(img, a, b, num=20, dataset_img_shape=(20, 20)):
    """
    Split table border with a and b coordinates of endings into num amount of square samples
    to find holes in.
    Returns np.array of the result images (reshaped with dataset_img_shape)
    and np.array of (x, y, r) for each sample image (where (x, y) is the sample image center coordinates
    and r is the sample image radius).
    """
    a = geom.Point(a)
    b = geom.Point(b)
    (n, m, _) = img.shape
    r = round(a.distance(b) / num)
    border_images = []
    border_squares = []
    for i in range(num // 10, num * 9 // 10):
        mid = (a * i + b * (num - i)) / num
        x, y = round(mid[0]), round(mid[1])
        if 0 <= y - r and y + r < n and 0 <= x - r and x + r < m:
            cropped = img[y-r:y+r, x-r:x+r, :]
            cropped = cv2.resize(cropped, dataset_img_shape, interpolation=cv2.INTER_AREA)
            border_images.append(cropped)
            border_squares.append((x, y, r))
    return np.array(border_images), np.array(border_squares)


def find_holes(model, img, table):
    """
    Takes hole_nn_model (can be constructed using hole_recognition.hole_nn_model.prepare_model()),
    image and billiard table coordinates on it.
    Returns 0 if table[0]..table[1] and table[2]..table[3] table borders are more likely to have holes in their
    midpoints than table[1]..table[2] and table[3]..table[0] do. Otherwise returns 1.
    """
    (n, m, _) = img.shape
    not_hole_prob = []
    k = 0.8  # probability that the model prediction is correct
    for i in range(len(table)):
        border_images, _ = split_border(img, table[i], table[i-1], num=60)
        one_side_hole_probs = model.predict(border_images)
        not_hole_prob.append((1 - one_side_hole_probs * k).prod())
    prob0 = not_hole_prob[0] * not_hole_prob[2]
    prob1 = not_hole_prob[1] * not_hole_prob[3]
    return 1 if prob1 > prob0 else 0
