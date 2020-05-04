import os
import cv2
import numpy as np
from PIL import Image
import datetime

solid_balls_cnt = 0
striped_balls_cnt = 0
not_balls_cnt = 0
N = 32


def process_image(img, solid_balls_dir, striped_balls_dir, not_balls_dir, img_path):
    global solid_balls_cnt
    global striped_balls_cnt
    global not_balls_cnt
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, minDist=15, dp=2,
                               param1=100,
                               param2=25,
                               minRadius=8,
                               maxRadius=20)
    circles = np.uint16(np.around(circles))
    (n, m, _) = img.shape

    def save_image(image, x, y, r, path):
        try:
            cropped = np.array(image)[y - r:y + r, x - r:x + r, :]
            Image.fromarray(np.uint8(cropped)).resize((N, N), Image.ANTIALIAS).save(path)
        except:
            pass

    def create_path(dir, i):
        return dir + '/{}_{}.jpg'.format(i, img_path)

    for i in circles[0, :]:
        x, y, r = i[0], i[1], i[2]
        new_img = img.copy()
        cv2.circle(new_img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(new_img, (x, y), 2, (0, 0, 255), 3)
        cv2.imshow('billiard', new_img)
        stop = False
        while True:
            key = cv2.waitKey(10)
            if key == ord('1'):  # solid ball
                solid_balls_cnt += 1
                path = create_path(solid_balls_dir, solid_balls_cnt)
                save_image(img, x, y, r, path)
                break
            if key == ord('2'):  # striped ball
                striped_balls_cnt += 1
                path = create_path(striped_balls_dir, striped_balls_cnt)
                save_image(img, x, y, r, path)
                break
            if key == ord('n'):  # not ball
                not_balls_cnt += 1
                path = create_path(not_balls_dir, not_balls_cnt)
                save_image(img, x, y, r, path)
                break
            if key == ord('s'):  # skip
                break
            if key == 27:
                stop = True
                break
        if stop:
            break
        cv2.destroyWindow('billiard')
    cv2.waitKey()


if __name__ == '__main__':
    def create_dir_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    images_dir = os.getcwd() + '/../../data/sync/images_for_dataset'

    tm = datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")
    dataset_dir = '../../data/sync/dataset_' + tm
    solid_balls_dir = dataset_dir + '/solid_balls'
    striped_balls_dir = dataset_dir + '/striped_balls'
    not_balls_dir = dataset_dir + '/not_balls'

    create_dir_if_not_exists(dataset_dir)
    create_dir_if_not_exists(solid_balls_dir)
    create_dir_if_not_exists(striped_balls_dir)
    create_dir_if_not_exists(not_balls_dir)

    for _, _, files in os.walk(images_dir):
        for file in sorted(files):
            img = cv2.imread(images_dir + '/' + file)
            process_image(
                img,
                solid_balls_dir,
                striped_balls_dir,
                not_balls_dir,
                os.path.splitext(file)[0]
            )
