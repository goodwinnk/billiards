import os
import cv2
import numpy as np
from PIL import Image


def process_image(img, balls_cnt, not_balls_cnt, out_balls_dir, out_not_balls_dir):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, minDist=15, dp=2,
                               param1=100,
                               param2=25,
                               minRadius=8,
                               maxRadius=20)
    circles = np.uint16(np.around(circles))
    (n, m, _) = img.shape
    for i in circles[0, :]:
        x, y, r = i[0], i[1], i[2]
        newimg = img.copy()
        cv2.circle(newimg, (x, y), r, (0, 255, 0), 2)
        cv2.circle(newimg, (x, y), 2, (0, 0, 255), 3)
        cv2.imshow('billiard', newimg)
        stop = False
        while True:
            key = cv2.waitKey(10)
            if key == ord('y'):  # ball
                try:
                    ball_img = np.array(img)[max(0, y - r):min(n, y + r), max(0, x - r):min(m, x + r), :]
                    balls_cnt += 1
                    filepath = out_balls_dir + '/{}.png'.format(balls_cnt)
                    print(filepath)
                    cropped = Image.fromarray(np.uint8(ball_img)).resize((20, 20), Image.ANTIALIAS)
                    print(cropped)
                    cropped.save(filepath)
                except:
                    pass
                break
            elif key == ord('n'):  # not ball
                try:
                    not_ball_img = np.array(img)[max(0, y - r):min(n, y + r), max(0, x - r):min(m, x + r), :]
                    not_balls_cnt += 1
                    filepath = out_not_balls_dir + '/{}.png'.format(not_balls_cnt)
                    cropped = Image.fromarray(np.uint8(not_ball_img)).resize((20, 20), Image.ANTIALIAS)
                    cropped.save(filepath)
                except:
                    pass
                break
            elif key == ord('s'):  # skip
                break
            elif key == 27:
                stop = True
                break
        if stop:
            break
        cv2.destroyWindow('billiard')
    cv2.waitKey()
    return balls_cnt, not_balls_cnt


if __name__ == '__main__':
    images_dir = os.getcwd() + '/../data/sync/images_for_dataset'
    balls_cnt = not_balls_cnt = 0
    for _, _, files in os.walk(images_dir):
        for file in sorted(files):
            img = cv2.imread(images_dir + '/' + file)
            balls_cnt, not_balls_cnt = process_image(
                img,
                balls_cnt,
                not_balls_cnt,
                os.getcwd() + '/../data/sync/dataset/balls',
                os.getcwd() + '/../data/sync/dataset/not_balls'
            )
