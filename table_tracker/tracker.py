import cv2
import sympy.geometry as geom
import numpy as np


class TableTracker:
    def __init__(self):
        self.trackers = cv2.MultiTracker_create()
        self.prev_state = None
        self.new_state = None
        self.error_occurred = False

    def update(self, frame):
        success, boxes = self.trackers.update(frame)
        if not success:
            self.error_occurred = True
        self.prev_state = self.new_state
        self.new_state = boxes

    def reset(self, frame, table):
        self.trackers = cv2.MultiTracker_create()
        self.prev_state = None
        self.new_state = []
        self.error_occurred = False

        a, b, c, d = tuple(map(geom.Point, table))
        (n, m, _) = frame.shape

        # box radius
        r = round((a.distance(b) + b.distance(c) + c.distance(d) + d.distance(a)) / (4 * 20))

        for corner in table:
            x, y = corner
            if 0 <= y - r and y + r < n and 0 <= x - r and x + r < m:
                box = (x, y, r, r)
                self.trackers.add(cv2.TrackerCSRT_create(), frame, box)
                self.new_state.append(box)

    def has_changed(self):
        if self.error_occurred:
            return True

        if self.new_state is None or self.prev_state is None:
            return False

        if len(self.new_state) == 0 or len(self.prev_state) == 0:
            return True

        prev = np.array(self.prev_state)
        new = np.array(self.new_state)
        return (np.abs(prev - new)).mean() > 3
