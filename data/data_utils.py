from pytube import YouTube
import os
import cv2


# Download youtube video
def download_youtube_video(url):
    return YouTube(url).streams.get_highest_resolution().download()


# Check video frame by frame
# 'f' for enabling frame-by-frame mode and for the next frame
# 'c' for exit from frame-by-frame mode
# 's' for saving current frame
def frame_by_frame_play(video_path: str, skip_seconds=0, stop_on_start=False):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video.set(cv2.CAP_PROP_POS_FRAMES, int(skip_seconds * fps))

    is_paused = stop_on_start

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        cv2.imshow("video", frame)

        if is_paused:
            while True:
                key = cv2.waitKey(0)
                if key == ord('c'):
                    is_paused = False
                    break
                elif key == ord('f'):
                    break
                elif key == ord('s'):
                    base_path = os.path.splitext(video_path)[0]
                    cur_index = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                    frame_path = base_path + "_" + str(cur_index) + ".jpg"
                    cv2.imwrite(frame_path, frame)
        else:
            key = cv2.waitKey(1)
            if key == ord('f'):
                is_paused = True

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


# extract_images_from_video("The Train exercise by Fedor Gorst.mp4", skip_seconds=6, max_count=50)
# frame_by_frame_play("Billiards Tutorial How to Break 8 Ball in Pool.mp4", skip_seconds=118, stop_on_start=True)

if __name__ == '__main__':
    frame_by_frame_play(
        download_youtube_video("https://www.youtube.com/watch?v=_xig92Lo72M"),
        skip_seconds=118,
        stop_on_start=True)
