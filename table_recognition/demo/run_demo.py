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

from data_utils.utils import download_youtube_video
from pathlib import Path
import subprocess
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

    subprocess.run([
        'python3', TABLE_RECOGNITION_ROOT / 'find_table_polygon.py',
        '--video', input_video_path,
        '--layout', layout_path
    ])

    print('Drawing table...')

    subprocess.run([
        'python3', TABLE_RECOGNITION_ROOT / 'highlight_table.py',
        '--video', input_video_path,
        '--layout', layout_path,
        '--output', video_with_table_path
    ])

    print(f'Video with highlighted table saved in: {os.path.abspath(video_with_table_path)}')


if __name__ == '__main__':

    if os.path.exists(GENERATED_PATH):
        shutil.rmtree(GENERATED_PATH)
    os.makedirs(GENERATED_PATH, exist_ok=False)

    print('Executing local data demo...')
    draw_table(LOCAL_INPUT_VIDEO_PATH, LOCAL_LAYOUT_PATH, LOCAL_VIDEO_WITH_TABLE_PATH)

    print('Executing youtube data demo...')
    if os.path.exists(YOUTUBE_INPUT_VIDEO_PATH):
        os.remove(YOUTUBE_INPUT_VIDEO_PATH)
    download_youtube_video(YOTUBE_DATA_URL, DEMO_ROOT / 'tmp')
    for f in os.listdir(DEMO_ROOT / 'tmp'):
        path = DEMO_ROOT / 'tmp' / Path(f)
        subprocess.run([
            'python3', ROOT / 'data_utils' / 'data_cli.py', 'cut',
            path, YOUTUBE_INPUT_VIDEO_PATH,
            '00:00:45', '00:00:50'
        ])
        shutil.rmtree(DEMO_ROOT / 'tmp')
    draw_table(YOUTUBE_INPUT_VIDEO_PATH, YOUTUBE_LAYOUT_PATH, YOUTUBE_VIDEO_WITH_TABLE_PATH)
