from itertools import takewhile

from cv2 import imshow, destroyWindow, namedWindow, WINDOW_AUTOSIZE

from data_utils.utils import frame_by_frame_play, download_youtube_video
from game_model.poolcv import PoolCV

if __name__ == '__main__':
    poolCV = PoolCV(ball_detect_net_path="../ball_detection/candidate_classifier/weights.pt",
                    hole_detect_net_path="../hole_recognition/weights.pt")

    model_window_name = "model"
    namedWindow(model_window_name, flags=WINDOW_AUTOSIZE)

    def frame_modifier(frame, index):
        poolCV.update(frame, index)
        poolCV.draw_game_on_image(frame, draw_net=False)

        for event in reversed(list(takewhile(lambda e: e.frame_index == index, reversed(poolCV.log)))):
            print(event)

        model_image = poolCV.get_model_image()
        imshow(model_window_name, model_image)

        return frame

    frame_by_frame_play(
        video_path=download_youtube_video(
            "https://www.youtube.com/watch?v=mF7iqCiHt1Y&t=84s",
            output_path="../data/local",
            filename="TOP_15_BEST_SHOTS_Mosconi_Cup_2019_9_ball_Pool.mp4"
        ),
        start_from_frame=330,
        frame_modifier=frame_modifier,
        stop_on_start=False
    )

    destroyWindow(model_window_name)
