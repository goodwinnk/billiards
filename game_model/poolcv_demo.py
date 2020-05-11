from cv2 import imshow, destroyWindow, namedWindow, WINDOW_AUTOSIZE

from data_utils.utils import frame_by_frame_play
from game_model.poolcv import PoolCV

if __name__ == '__main__':
    poolCV = PoolCV(ball_detect_net_path="../ball_detection/candidate_classifier/weights.pt")

    model_window_name = "model"
    namedWindow(model_window_name, flags=WINDOW_AUTOSIZE)

    def frame_modifier(frame, index):
        poolCV.update(frame, index)
        poolCV.draw_game_on_image(frame, draw_net=False)

        model_image = poolCV.get_model_image()
        imshow(model_window_name, model_image)

        return frame

    frame_by_frame_play(
        video_path="../data/sync/poolcv_demo/20200120_115235-resized.m4v",
        start_from_frame=570,
        frame_modifier=frame_modifier,
        stop_on_start=False
    )

    destroyWindow(model_window_name)
