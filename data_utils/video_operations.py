import typing

import cv2
import numpy as np


# Writes frames into writer one by one, but it can requier a lot of memory
# It is not recommended to use this function for saving big videos
def save_frames_as_video(path, frames: typing.List[np.ndarray], fps):
    h, w = frames[0].shape[: 2]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f'{path} saved')
    print(f'Number of frames: {len(frames)}')
    print(f'fps = {fps}')
    if len(frames) > 0:
        print(f'Resolution: {frames[0].shape}')
