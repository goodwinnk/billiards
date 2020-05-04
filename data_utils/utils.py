import os
from datetime import timedelta
from typing import Optional

import cv2
import dateparser
import numpy as np
from pytube import YouTube

from youtube_dl import YoutubeDL

import subprocess


# Downloads all videos from youtube channel, takes n_frames uniform frames from each video and saves them into
# structured directories (channel_id/video_id/*.png)
def download_frame_from_yotube_video(channel_url, n_frames, output_path):
    with YoutubeDL({'format': 'mp4'}) as ydl:
        info_dict = ydl.extract_info(channel_url, download=False)
        channel_id = info_dict.get('id')
        for video in info_dict.get('entries'):
            video_id = video.get('id')
            video_duration = video.get('duration')
            video_url = video.get('url')
            dir_path = os.path.join(output_path, f'channel_{channel_id}', f'video_{video_id}')
            os.makedirs(dir_path, exist_ok=True)
            for i in range(n_frames):
                file_path = os.path.join(dir_path, f'{1 + len(os.listdir(dir_path))}.png')
                timestamp = int((i + 1) * video_duration / (n_frames + 1))
                subprocess.run([
                    'ffmpeg', '-hide_banner', '-loglevel', 'panic',
                    '-ss', str(timestamp),
                    '-i', video_url,
                    '-vframes', '1',
                    '-q:v', '2',
                    file_path
                ])


# Download youtube video
def download_youtube_video(url, output_path: Optional[str] = None):
    return YouTube(url).streams.get_highest_resolution().download(output_path)


# Check video frame by frame
# 'f' enable frame-by-frame mode and go to the next frame
# 'd' enable frame-by-frame mode and go to the previous frame
# 'c' exit from frame-by-frame mode
# 's' save current frame
def frame_by_frame_play(
        video_path: str,
        skip_seconds=0,
        start_from_frame: Optional[int] = None,
        stop_on_start=False,
        frame_save_modifier="mod",
        frame_output_path: Optional[str] = None,
        frame_modifier=lambda frame, index: frame):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    base_frame_path = os.path.splitext(video_path)[0] if frame_output_path is None \
        else os.path.join(frame_output_path, "frame")

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.set(cv2.CAP_PROP_POS_FRAMES, int(skip_seconds * fps))

    if start_from_frame is not None:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_from_frame)

    is_paused = stop_on_start
    is_origin_frame = False

    video_window = "video"
    cv2.namedWindow(video_window, cv2.WINDOW_AUTOSIZE)

    def is_video_closed():
        return cv2.getWindowProperty(video_window, cv2.WINDOW_AUTOSIZE) == -1

    while not is_video_closed():
        frame_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))

        ret, frame = video.read()
        if not ret:
            break

        modified_frame = frame
        if not is_origin_frame:
            modified_frame = frame_modifier(frame, frame_index)

        cv2.rectangle(modified_frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(modified_frame, str(frame_index), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow(video_window, modified_frame)

        if is_paused:
            while True:
                if is_video_closed():
                    break

                key = cv2.waitKey(10)
                if key == ord('c'):
                    is_paused = False
                    break
                elif key == ord('f'):
                    break
                elif key == ord('d'):
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
                    break
                elif key == ord('o'):
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    is_origin_frame = not is_origin_frame
                    break
                elif key == ord('s'):
                    frame_suffix = frame_save_modifier if not is_origin_frame else ""
                    frame_path = base_frame_path + "_" + frame_suffix + str(frame_index) + ".jpg"
                    cv2.imwrite(frame_path, modified_frame)
                elif key == 27:
                    cv2.destroyWindow(video_window)
                    break
        else:
            key = cv2.waitKey(1)
            if key == ord('f') or key == ord('d'):
                is_paused = True
            if key == ord('o'):
                is_paused = True
                is_origin_frame = not is_origin_frame
            elif key == 27:
                cv2.destroyWindow(video_window)

    video.release()


# Extract frames from video
def extract_images_from_video(video_path, out_path="frames", delay_seconds=0.5, skip_seconds=0, max_count: int = None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    if os.path.exists(out_path):
        raise FileExistsError(out_path)

    os.mkdir(out_path)

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frame_index = int(skip_seconds * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    count = 0

    while max_count is not None and count < max_count:
        success, image = video.read()
        if not success:
            break

        count += 1

        cv2.imwrite(os.path.join(out_path, "frame_" + str(count) + ".jpg"), image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        cv2.imwrite(os.path.join(out_path, "frame_gray_" + str(count) + ".jpg"), gray)

        frame_index += int(delay_seconds * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    video.release()


def play_with_move_detect(video_path: str, skip_seconds=0, stop_on_start=False):
    sub = cv2.createBackgroundSubtractorKNN()

    def to_gray_gaussian(frame, index):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_gray = cv2.GaussianBlur(gray, (21, 21), 0)

        return sub.apply(gaussian_gray)

    frame_by_frame_play(video_path,
                        frame_modifier=to_gray_gaussian,
                        stop_on_start=stop_on_start,
                        skip_seconds=skip_seconds)


# Takes time in format hh:mm:ss and returns equal time in seconds
def parse_seconds_from_hh_mm_ss(hh_mm_ss):
    d = dateparser.parse(hh_mm_ss)
    return timedelta(hours=d.hour, minutes=d.minute, seconds=d.second).total_seconds()


# from_s: left bound in seconds
# to_s: right bound in seconds
def cut_video(video_path: str, out_path: str, from_s: float, to_s: float):
    capture = cv2.VideoCapture(video_path)
    fps = np.round(capture.get(cv2.CAP_PROP_FPS))
    from_frame = int(from_s * fps)
    to_frame = int(to_s * fps)

    writer = None

    capture.set(cv2.CAP_PROP_POS_FRAMES, from_frame)
    frame_id = from_frame

    while capture.isOpened():
        response, frame = capture.read()
        if not response or frame_id > to_frame:
            break
        if writer is None:
            h, w = frame.shape[: 2]
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(frame)
        frame_id += 1

    if writer is not None:
        writer.release()


if __name__ == '__main__':
    play_with_move_detect(
        download_youtube_video("https://www.youtube.com/watch?v=_xig92Lo72M", "../data/local/data_utils_demo"),
        skip_seconds=119, stop_on_start=True
    )
    # frame_by_frame_play(
    #     download_youtube_video("https://www.youtube.com/watch?v=_xig92Lo72M"),
    #     skip_seconds=119,
    #     stop_on_start=True)
