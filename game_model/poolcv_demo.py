from data_utils.utils import frame_by_frame_play
from game_model.poolcv import PoolCV

if __name__ == '__main__':
    poolCV = PoolCV(ball_detect_net_path="../ball_detection/candidate_classifier/weights.pt")

    def frame_modifier(frame, index):
        poolCV.update(frame, index)
        poolCV.draw_game_on_image(frame)
        return frame

    frame_by_frame_play(
        video_path="../data/sync/poolcv_demo/20200120_115235-resized.m4v",
        start_from_frame=570,
        frame_modifier=frame_modifier)

