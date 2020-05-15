import cv2
import os
from table_recognition.find_table_polygon import find_table_layout_on_frame
from table_recognition.highlight_table import highlight_table_on_frame
from hole_recognition.process_holes import split_border
import sympy.geometry as geom
import datetime

if __name__ == '__main__':
    def create_dir_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


    tm = datetime.datetime.now().strftime("%H:%M:%S_%Y-%m-%d")
    dataset_dir = '../data/sync/holes_dataset'
    images_dir = dataset_dir + '/images'
    holes_dir = dataset_dir + '/holes' + tm
    not_holes_dir = dataset_dir + '/not_holes' + tm
    
    create_dir_if_not_exists(holes_dir)
    create_dir_if_not_exists(not_holes_dir)

    for _, _, files in os.walk(images_dir):
        for file in sorted(files):
            filename = file.split('.')[0]
            img = cv2.imread(images_dir + '/' + file)
            (n, m, _) = img.shape
            table_img = img.copy()
            table = find_table_layout_on_frame(table_img)
            highlight_table_on_frame(table_img, table)

            N = 20
            for i in range(len(table)):
                a = geom.Point(table[i])
                b = geom.Point(table[i - 1])
                border_images, border_squares = split_border(img, a, b)
                for j in range(len(border_images)):
                    sample_image = table_img.copy()
                    x, y, r = border_squares[j]
                    cv2.circle(sample_image, (x, y), r, (0, 255, 0), thickness=1)
                    cv2.imshow('{}_{}_{}'.format(filename, i, j), sample_image)
                    while True:
                        key = cv2.waitKey(10)
                        if key == ord('y'):  # hole
                            cropped = border_images[j]
                            cv2.imwrite(holes_dir + '/{}_{}_{}.jpg'.format(filename, i, j), cropped)
                            break
                        if key == ord('n'):  # not hole
                            cropped = border_images[j]
                            cv2.imwrite(not_holes_dir + '/{}_{}_{}.jpg'.format(filename, i, j), cropped)
                            break
                        if key == ord('s'):
                            break
                    cv2.destroyAllWindows()
