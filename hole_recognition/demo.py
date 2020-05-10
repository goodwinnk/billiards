from hole_recognition.hole_nn_model import HoleDetector
from hole_recognition.process_holes import find_holes, split_border
from table_recognition.find_table_polygon import find_table_layout_on_frame
from table_recognition.highlight_table import highlight_table_on_frame
import cv2
import numpy as np
import os


def mark_hole_table_borders_demo(images_path, result_path, model):
    print('Running demo: mark hole table borders')

    for _, _, files in os.walk(images_path):
        for file in sorted(files):
            if file.split('.')[-1] != 'jpg':
                continue

            print('Process {}...'.format(file))
            img = cv2.imread(images_path + '/' + file)

            table = find_table_layout_on_frame(img)
            demo_img = img.copy()
            highlight_table_on_frame(demo_img, table)

            if find_holes(model, img, table) == 1:
                table = np.roll(table, 1, axis=0)

            for i in range(0, len(table), 2):
                cv2.line(demo_img, tuple(table[i]), tuple(table[i + 1]), (255, 0, 0), 4)

            cv2.imwrite(result_path + '/' + file, demo_img)


def mark_hole_probabilies_demo(images_path, result_path, model):
    print('Running demo: mark hole probabilities on table borders')

    for _, _, files in os.walk(images_path):
        for file in sorted(files):
            if file.split('.')[-1] != 'jpg':
                continue

            print('Process {}...'.format(file))
            img = cv2.imread(images_path + '/' + file)

            table = find_table_layout_on_frame(img)

            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1.0
            thickness = 2

            def get_text_start_point(center_point, text):
                center_point_x, center_point_y = center_point
                text_sz, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_sz_x, text_sz_y = text_sz
                return (center_point_x - text_sz_x // 2,
                        center_point_y + text_sz_y // 2)

            demo_img = img.copy()
            highlight_table_on_frame(demo_img, table)
            for i in range(len(table)):
                a, b = table[i], table[i-1]
                border_images, border_squares = split_border(img, a, b, num=60)
                probs = model.predict(border_images)

                for j in range(len(border_squares)):
                    sq = border_squares[j]
                    prob = probs[j][1]
                    if prob < 0.01:
                        continue
                    x, y, r = sq
                    label = str(round(10000 * prob) / 100) + '%'
                    cv2.putText(demo_img, label, get_text_start_point((x, y), label),
                                font, thickness=thickness, color=[0, 0, 0], fontScale=font_scale)
            cv2.imwrite(result_path + '/' + file, demo_img)


if __name__ == '__main__':
    demo_images_path = '../data/sync/holes_recognition_demo/images'
    demo1_result_path = '../data/sync/holes_recognition_demo/table_borders_result'
    demo2_result_path = '../data/sync/holes_recognition_demo/probabilities_result'
    m = HoleDetector()
    m.load()
    mark_hole_probabilies_demo(demo_images_path, demo2_result_path, m)
    mark_hole_table_borders_demo(demo_images_path, demo1_result_path, m)
