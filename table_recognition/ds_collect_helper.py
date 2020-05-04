import argparse
import os
import time
from tkinter import *

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

ACCEPTABLE_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg']


def is_image_format(path):
    for format in ACCEPTABLE_IMAGE_FORMATS:
        if str(path).endswith(format):
            return True
    return False


class Application:

    def __init__(self, width, height, layout_path, dir_path):
        self.t0 = time.time()

        self.width = width
        self.height = height

        self.layout_path = layout_path
        self.layout_dir_path = os.path.dirname(layout_path)
        self.file = open(layout_path, 'a')
        self.images_paths = set(os.path.relpath(os.path.join(dir_path, f), self.layout_dir_path)
                                for f in os.listdir(dir_path) if is_image_format(f))
        try:
            self.df = pd.read_csv(self.layout_path, header=None)
        except pd.errors.EmptyDataError:
            self.df = pd.DataFrame(columns=list(range(10)))
            for i in range(10):
                self.df[i] = []
                self.df[i] = self.df[i].astype(int if i > 0 else str)
        self.images_paths = list(self.images_paths.difference(set(self.df[0])))
        self.current_image_ptr = 0

        self.root = Tk()
        self.root.title('Table recognition ds helper')
        # self.root.attributes('-fullscreen', True)
        self.root.bind('q', self.close_button_clicked)
        self.root.bind('d', self.next_button_clicked)
        self.root.bind('r', self.flip_pocket_flag)

        # configure draw area
        self.canvas = Canvas(self.root, bg='white')
        self.canvas.pack(fill=BOTH, expand=YES)
        self.canvas.bind('<B1-Motion>', self.rmouse_clicked)
        self.d = 0
        self.img = None

        self.polygon_vertices = None
        self.pocket_flag = None

        self.set_image(self.current_image_ptr)

    def reset_polygon(self):
        self.polygon_vertices = np.array(
            [[self.width // 3, self.height // 3], [2 * self.width // 3, self.height // 3],
             [2 * self.width // 3, 2 * self.height // 3], [self.width // 3, 2 * self.height // 3]])
        self.pocket_flag = 0

    def set_image(self, ptr):
        self.reset_polygon()
        title = f'{ptr+1}/{len(self.images_paths)}: {self.images_paths[ptr]}'
        self.root.title(title)
        self.draw()

    def flip_pocket_flag(self, event):
        self.pocket_flag ^= 1
        self.draw()

    def set_window_size(self, width, height):
        self.root.geometry(f'{width}x{height}')

    def find_nearest_vertex_id(self, x, y):
        p = np.array([x, y])
        best_id = 0
        best_dist = np.inf
        for id, q in enumerate(self.polygon_vertices):
            tmp_dist = np.linalg.norm(q - p)
            if tmp_dist < best_dist:
                best_dist, best_id = tmp_dist, id
        return best_id

    def rmouse_clicked(self, event):
        id = self.find_nearest_vertex_id(event.x, event.y)
        self.polygon_vertices[id] = np.array([event.x, event.y])
        self.draw()

    def draw(self):
        self.canvas.delete('all')

        np_img = cv2.imread(os.path.abspath(os.path.join(self.layout_dir_path,
                                                         self.images_paths[self.current_image_ptr])))
        self.img = ImageTk.PhotoImage(Image.fromarray(np_img[:, :, ::-1]))

        h, w = np_img.shape[: 2]
        self.d = int(0.25 * max(h, w))

        self.canvas.create_image(self.d, self.d, image=self.img, anchor=NW)
        self.set_window_size(2 * self.d + w, 2 * self.d + h)

        self.canvas.create_polygon(self.polygon_vertices.reshape(-1).tolist(), fill='', outline='green', width=2.5)

        r = 10
        for i, (x, y) in enumerate(self.polygon_vertices):
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='', outline='red', width=2)
            if i % 2 != self.pocket_flag:
                j = (i + 1) % len(self.polygon_vertices)
                qx, qy = (self.polygon_vertices[j] + self.polygon_vertices[i]) / 2
                self.canvas.create_oval(qx - r, qy - r, qx + r, qy + r, fill='', outline='red', width=2)

        self.canvas.create_text(0, 0, anchor='nw', fill='darkblue', font='Times 20 italic bold', text=str(time.time() - self.t0))

    def close_button_clicked(self, event=None):
        self.root.quit()

    # Considers table vertices as pairs and sorts them, takes first half of them and returns point with smallest
    # seconds coordinate
    def get_up_half_left_point(self):
        ids = list(range(len(self.polygon_vertices)))
        ids.sort(key=lambda i: tuple(self.polygon_vertices[i]))
        print(self.polygon_vertices[ids])
        if self.polygon_vertices[ids[0]][0] < self.polygon_vertices[ids[1]][0]:
            return ids[0]
        else:
            return ids[1]

    def get_relative_hull_with_pocket_flag(self):
        hull = []
        i = self.get_up_half_left_point()
        v = np.array([self.d, self.d])
        n = len(self.polygon_vertices)
        for iter in range(n):
            j = (iter + i) % n
            hull.append(self.polygon_vertices[j] - v)
        return np.array(hull), int(self.pocket_flag ^ (i % 2))

    def next_button_clicked(self, event=None):
        rel_hull, rel_flag = self.get_relative_hull_with_pocket_flag()
        print(type(rel_hull), type(rel_flag))
        record = pd.DataFrame([[self.images_paths[self.current_image_ptr]]
                               + rel_hull.reshape(-1).tolist()
                               + [rel_flag]])
        print(record)
        self.df = self.df.append(record, ignore_index=True)
        self.df.to_csv(self.layout_path, index=False, header=None)
        print(self.df)
        self.current_image_ptr += 1
        if self.current_image_ptr == len(self.images_paths):
            self.root.quit()
        self.set_image(self.current_image_ptr)

    def start(self):
        self.root.mainloop()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Table recognition ds collecting helper')
    parser.add_argument('--dir', help='Path to the directory with images', required=True)
    parser.add_argument('--layout', help='Path to the file with table layout', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    app = Application(width=1080, height=720, layout_path=args.layout, dir_path=args.dir)
    app.start()
