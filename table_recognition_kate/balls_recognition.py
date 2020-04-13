import cv2
from queue import Queue
import numpy as np


def bfs(img, used, start, predicate=lambda x, y: True, action=lambda x, y: None):
    """
    Visits non-used img pixels (from start pixel) which satisfy predicate,
    marks all visited pixels as used=True and calls action function on
    each pixel with two arguments, the pixel coordinates.
    """
    (n, m, _) = img.shape
    (i, j) = start
    if not predicate(i, j):
        return
    used[i][j] = True
    q = Queue()
    q.put((i, j))
    while not q.empty():
        (x, y) = q.get()
        action(x, y)
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and not used[nx][ny] and predicate(nx, ny):
                used[nx][ny] = True
                q.put((nx, ny))


def largest_component(img, predicate=lambda x, y: True):
    """
    finds the largest component in image matrix, where each pixel satisfies the
    predicate and returns the numpy array of all pixels coordinates.
    """
    (n, m, _) = img.shape
    used = np.zeros((n, m), dtype=bool)
    res = []
    for i in range(n):
        for j in range(m):
            if not used[i][j]:
                # mark new component
                component = []

                def action(x, y):
                    component.append((x, y))

                bfs(img, used, (i, j), predicate, action)

                if len(res) < len(component):
                    res = component

    return np.array(res)


def mark_balls(frame, index):
    (n, m, _) = frame.shape

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def is_table(x: int, y: int):
        pixel = hsv_frame[x][y]
        nearby = hsv_frame[max(0, x - 5):min(n, x + 5) + 1, max(0, y - 5):min(m, y + 5) + 1, 0::2].reshape((-1, 2))
        return (70 < pixel[0] < 170 or 190 < pixel[0] < 290 or pixel[2] < 20) and pixel[1] > 50 and \
               np.std(nearby, axis=0).mean() < 10

    table = largest_component(frame, is_table)

    img = frame.copy()

    def fill(cmp, pixel):
        for x, y in cmp:
            img[x, y] = pixel

    fill(table, [255, 255, 255])

    used = np.zeros((n, m), dtype=bool)
    for (x, y) in table:
        used[x, y] = True

    for i in range(n):
        for j in range(m):
            if not used[i][j]:
                comp = []
                has_bound = False

                def action(x: int, y: int):
                    comp.append((x, y))
                    nonlocal has_bound
                    if x == 0 or y == 0 or x == n - 1 or y == m - 1:
                        has_bound = True

                bfs(frame, used, (i, j), action=action)

                if has_bound:
                    fill(comp, [255, 255, 255])
                else:
                    fill(comp, [0, 0, 0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, minDist=15, dp=1.1, param1=100, param2=10, minRadius=13,
                               maxRadius=30)

    res = frame.copy()
    for x, y, r in circles[0, :]:
        cv2.circle(res, (x, y), r, (255, 0, 100), 2)

    return res


if __name__ == '__main__':
    for i in range(1, 6):
        print(i)
        img = cv2.imread(f'../../example{i}.jpg')
        out_img = mark_balls(img, None)
        cv2.imwrite(f'../../example{i}_res.jpg', out_img)

