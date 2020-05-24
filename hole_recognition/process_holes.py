import sympy.geometry as geom
import numpy as np
import cv2


def get_model_input(img, x, y, r, inp_img_shape=(20, 20)):
    cropped = img[y - r:y + r, x - r:x + r, :]
    return cv2.resize(cropped, inp_img_shape, interpolation=cv2.INTER_AREA)


def split_border(img, a, b, num=20):
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
    r = max(round(a.distance(b) / num), 1)
    border_images = []
    border_squares = []
    for i in range(num // 10, num * 9 // 10):
        mid = (a * i + b * (num - i)) / num
        x, y = round(mid[0]), round(mid[1])
        if 0 <= y - r and y + r < n and 0 <= x - r and x + r < m:
            border_images.append(get_model_input(img, x, y, r))
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

        if len(border_images) == 0:
            not_hole_prob.append(0.5)
            continue

        one_side_hole_probs = model.predict(border_images)
        not_hole_prob.append((1 - one_side_hole_probs * k).prod().item())
    prob0 = not_hole_prob[0] * not_hole_prob[2]
    prob1 = not_hole_prob[1] * not_hole_prob[3]
    return 1 if prob1 > prob0 else 0


def rotate_table(model, img, table):
    """
    Rotates the table corners (returned by find_table_layout_on_frame)
    so that the holes are at table[1]..table[2], table[3]..table[0] sides and
    table[0] has the least y-value possible.
    """
    corners_order = find_holes(model, img, table)
    if corners_order == 1:
        table = np.roll(table, 1, axis=0)
    if table[2][1] > table[0][1]:
        table = np.roll(table, 2, axis=0)
    return table


def check_table_corners(model, img, table):
    """
    Checks the correctness of the found table by checking the corner holes using the hole recognition model.
    Returns True if the table is most probably correct, False otherwise.
    """
    a, b, c, d = tuple(map(geom.Point, table))
    (n, m, _) = img.shape
    r = round((a.distance(b) + b.distance(c) + c.distance(d) + d.distance(a)) / (4 * 20))
    samples = []
    for v in table:
        x, y = v
        if 0 <= y - r and y + r < n and 0 <= x - r and x + r < m:
            sample = get_model_input(img, x, y, r)
            samples.append(sample)
    if len(samples) == 0:
        return True
    samples = np.array(samples)
    probs = model.predict(samples)  # the probabilities of holes for each table corner
    probs = np.array(probs.detach())
    probs = probs[:, 1:]
    print(probs)
    return np.count_nonzero(probs > 0.8) > 0.5 * len(probs)
