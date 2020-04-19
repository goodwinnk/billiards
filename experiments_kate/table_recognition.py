import numpy as np
import sympy as sp
import cv2
from queue import Queue
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


# bfs visits non-used img pixels which satisfy predicate and calls
# action function with two arguments, the coordinates of pixel.
def bfs(img, used, start, predicate, action):
    (n, m, _) = img.shape
    (i, j) = start
    if not predicate(img[i][j]):
        return
    used[i][j] = True
    q = Queue()
    q.put((i, j))
    while not q.empty():
        (x, y) = q.get()
        action(x, y)
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and not used[nx][ny] and predicate(img[nx][ny]):
                used[nx][ny] = True
                q.put((nx, ny))


# is_table_color function returns True if the given pixel has the same
# color as the billiard table might have.
def is_table_color(pixel):
    pixel = cv2.cvtColor(np.array([[pixel]]), cv2.COLOR_RGB2HSV)[0][0]
    return 70 < pixel[0] < 270 and pixel[1] > 70 and pixel[2] > 80


# finds the largest component in image, where each pixel satisfies the
# predicate and returns the numpy array of all pixels coordinates.
def largest_component(img_matrix, predicate):
    (n, m, _) = img_matrix.shape
    used = np.zeros((n, m), dtype=bool)
    res = []
    for i in range(n):
        for j in range(m):
            if not used[i][j]:
                # mark new component
                component = []

                def action(x, y):
                    component.append((x, y))

                bfs(img_matrix, used, (i, j), is_table_color, action)

                if len(res) < len(component):
                    res = component

    return np.array(res)


# deletes similar (satisfying is_similar predicate) neightbouring array
# elements and returns the array with numpy.mean element for each
# equivalence class. Raises an error if the result size is greater than
# max_res_size.
def unique(array, is_similar, max_res_size):
    classes = []
    for i in range(len(array)):
        found = False
        for j in range(len(classes)):
            if is_similar(array[i], classes[j][0]):
                classes[j].append(array[i])
                found = True
                break
        if not found:
            classes.append([array[i]])
            if len(classes) > max_res_size:
                raise ValueError('More than {} equivalence classes'.format(max_res_size))

    return [np.mean(c, axis=0) for c in classes]


# table_coordinates tries to find a billiard table in input_image_path,
# saves the image with found lines in output_image_path (if it is not None)
# and returns 4 lines - borders of the billiard table.
# May throw an exception if it can't recognize the table.
def table_coordinates(input_image_path, output_image_path=None):
    with Image.open(input_image_path) as img:
        img_matrix = np.array(img)
    (n, m, _) = img_matrix.shape

    # fing the largest component of table-colour neighboring pixels in img
    table = largest_component(img_matrix, is_table_color)

    # find convex hull of the largest component and draw it in hull_img
    hull_img = Image.new(mode='RGB', size=(m, n))
    draw = ImageDraw.Draw(hull_img)
    for simplex in ConvexHull(table).simplices:
        p1, p2 = table[simplex][0], table[simplex][1]
        draw.line(tuple(p1[::-1]) + tuple(p2[::-1]), fill=(255, 255, 255))

    # perform Hough transform
    gray = cv2.cvtColor(np.array(hull_img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is None:
        raise ValueError('Can\'t recognize the table.')

    # delete similar lines in found lines list
    def hough_to_sympy(x):
        r, theta = x[0][0], x[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0, y0 = a * r, b * r
        return sp.Line(sp.Point(x0 + 2000. * b, y0 - 2000. * a),
                       sp.Point(x0 - 2000. * b, y0 + 2000. * a))

    def lines_equal(x, y):
        x = hough_to_sympy(x)
        y = hough_to_sympy(y)
        ints = x.intersection(y)
        if len(ints) == 0:
            return False
        if not isinstance(ints[0], sp.Point2D):
            return True
        p = (float(ints[0][0]), float(ints[0][1]))
        return max(abs(2 * p[0] / m - 1), 1) * \
               max(abs(2 * p[1] / n - 1), 1) * \
               abs(float(x.angle_between(y))) < 0.2

    # fix this
    lines = unique(lines, lines_equal, 6)
    lines = unique(lines, lines_equal, 4)

    if len(lines) != 4:
        raise ValueError('Can\'t recognize the table.')

    # convert Hough lines to sympy.Line
    sp_lines = []
    for line in lines:
        sp_lines.append(hough_to_sympy(line))

    if output_image_path is not None:
        table_img = Image.fromarray(img_matrix)
        draw = ImageDraw.Draw(table_img)
        for line in sp_lines:
            draw.line(tuple(line.p1) + tuple(line.p2), fill=255, width=4)
        table_img.save(output_image_path)

    return lines


if __name__ == '__main__':
    for i in range(1, 14):
        inp = '../data/sync/experiments_kate/table_recognition/billiard_sample{}.jpg'.format(i)
        outp = '../data/sync/experiments_kate/table_recognition/billiard_sample{}_result.jpg'.format(i)
        try:
            table_coordinates('../data/sync/experiments_kate/table_recognition/billiard_sample{}.jpg'.format(i))
        except:
            pass
