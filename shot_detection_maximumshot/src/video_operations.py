import numpy as np
import cv2


def save_frames_as_video(path, frames, fps):
    h, w = frames[0].shape[: 2]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f'{path} saved')
