import os
import sys

PYTHONPATH = 'PYTHONPATH'
if PYTHONPATH not in os.environ:
    os.environ[PYTHONPATH] = ''
if len(os.environ[PYTHONPATH]) > 0:
    os.environ[PYTHONPATH] += f':{os.path.abspath(os.path.join("..", ".."))}'
else:
    os.environ[PYTHONPATH] = os.path.abspath(os.path.join('..', '..'))
sys.path.append(os.environ[PYTHONPATH])

from data_utils.utils import download_youtube_video, cut_video, parse_seconds_from_hh_mm_ss
from table_recognition.highlight_table import highlight_table
from table_recognition.find_table_polygon import find_table_layout
from pathlib import Path
import shutil


ROOT = Path('..') / Path('..')
DATA_ROOT = ROOT / Path('data')
SYNC_ROOT = DATA_ROOT / Path('sync')
LOCAL_ROOT = DATA_ROOT / 'local'
DEMO_ROOT = SYNC_ROOT / 'table_recognition_demo'
LOCAL_INPUT_VIDEO_PATH = DEMO_ROOT / 'local_demo_video.mp4'
YOUTUBE_INPUT_VIDEO_PATH = DEMO_ROOT / 'youtube_demo_video.mp4'
GENERATED_PATH = LOCAL_ROOT / 'table_recognition_demo_gen'
LOCAL_LAYOUT_PATH = GENERATED_PATH / 'local_layout.txt'
YOUTUBE_LAYOUT_PATH = GENERATED_PATH / 'youtube_layout.txt'
LOCAL_VIDEO_WITH_TABLE_PATH = GENERATED_PATH / 'local_video_table.mp4'
YOUTUBE_VIDEO_WITH_TABLE_PATH = GENERATED_PATH / 'youtube_video_table.mp4'
TABLE_RECOGNITION_ROOT = ROOT / 'table_recognition'
YOTUBE_DATA_URL = 'https://www.youtube.com/watch?v=YzqJxDx2Crc'


def draw_table(input_video_path, layout_path, video_with_table_path):
    print('Finding frame by frame table polygon...')
    find_table_layout(input_video_path, layout_path)
    print('Drawing table...')
    highlight_table(input_video_path, layout_path, video_with_table_path)
    print(f'Video with highlighted table saved in: {os.path.abspath(video_with_table_path)}')


if __name__ == '__main__':

    if os.path.exists(GENERATED_PATH):
        shutil.rmtree(GENERATED_PATH)
    os.makedirs(GENERATED_PATH, exist_ok=False)

    print('Executing local data demo...')
    draw_table(str(LOCAL_INPUT_VIDEO_PATH), str(LOCAL_LAYOUT_PATH), str(LOCAL_VIDEO_WITH_TABLE_PATH))

    print('Executing youtube data demo...')
    if os.path.exists(YOUTUBE_INPUT_VIDEO_PATH):
        os.remove(YOUTUBE_INPUT_VIDEO_PATH)
    download_youtube_video(str(YOTUBE_DATA_URL), str(DEMO_ROOT / 'tmp'))
    for f in os.listdir(DEMO_ROOT / 'tmp'):
        path = DEMO_ROOT / 'tmp' / Path(f)
        cut_video(
            str(path), str(YOUTUBE_INPUT_VIDEO_PATH),
            parse_seconds_from_hh_mm_ss('00:00:45'),
            parse_seconds_from_hh_mm_ss('00:00:50')
        )
        shutil.rmtree(DEMO_ROOT / 'tmp')
    draw_table(str(YOUTUBE_INPUT_VIDEO_PATH), str(YOUTUBE_LAYOUT_PATH), str(YOUTUBE_VIDEO_WITH_TABLE_PATH))
