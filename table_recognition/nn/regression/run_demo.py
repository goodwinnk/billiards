import os
import shutil
import sys
import zipfile
from time import time

PYTHONPATH = 'PYTHONPATH'
if PYTHONPATH not in os.environ:
    os.environ[PYTHONPATH] = ''
if len(os.environ[PYTHONPATH]) > 0:
    os.environ[PYTHONPATH] += f':{os.path.abspath(os.path.join("..", "..", ".."))}'
else:
    os.environ[PYTHONPATH] = os.path.abspath(os.path.join('..', '..', '..'))
sys.path.append(os.environ[PYTHONPATH])

from data_utils.utils import parse_seconds_from_hh_mm_ss, cut_video, download_youtube_video, get_video_resolution
from table_recognition.highlight_table import highlight_table
from table_recognition.find_table_polygon import buffered_find_table_layout
from table_recognition.nn.augm import compose, resize_transforms, hard_transforms, post_transforms, pre_transforms
import torch
from torch.utils.data import DataLoader
from table_recognition.nn.regression.train import RegressionFromSegmentation
from table_recognition.nn.table_recognition_ds import VideoTableRecognitionDataset
from table_recognition.table_recognition_models import NNRegressionBasedTableRecognizer
from pathlib import Path
import numpy as np

ROOT = Path('.') / '..' / '..' / '..'
DATA_ROOT = ROOT / Path('data')
SYNC_ROOT = DATA_ROOT / Path('sync')
LOCAL_ROOT = DATA_ROOT / 'local'
DEMO_ROOT = SYNC_ROOT / 'table_recognition_demo'
LOCAL_INPUT_VIDEO_PATH = DEMO_ROOT / 'local_demo_video.mp4'
YOUTUBE_INPUT_VIDEO_PATH = DEMO_ROOT / 'youtube_demo_video.mp4'
GENERATED_PATH = LOCAL_ROOT / 'table_recognition_demo_gen' / 'nn' / 'regression'
LOCAL_LAYOUT_PATH = GENERATED_PATH / 'local_layout.txt'
YOUTUBE_LAYOUT_PATH = GENERATED_PATH / 'youtube_layout.txt'
LOCAL_VIDEO_WITH_TABLE_PATH = GENERATED_PATH / 'local_video_table.mp4'
YOUTUBE_VIDEO_WITH_TABLE_PATH = GENERATED_PATH / 'youtube_video_table.mp4'
TABLE_RECOGNITION_ROOT = ROOT / 'table_recognition'
# YOTUBE_DATA_URL = 'https://www.youtube.com/watch?v=YzqJxDx2Crc'
# YOTUBE_DATA_URL = 'https://www.youtube.com/watch?v=kGz87hmDitg'
YOTUBE_DATA_URL = 'https://www.youtube.com/watch?v=w7XojtPAJOo'

ZIP_NN_PATH = SYNC_ROOT / 'table_recognition_regression_trained_model.zip'
REL_MODEL_PATH = Path('save') / 'best_model.pth'
MODEL_PATH = GENERATED_PATH / REL_MODEL_PATH


def draw_table(input_video_path, layout_path, video_with_table_path):
    print('Extracting neural network...')

    with zipfile.ZipFile(ZIP_NN_PATH, 'r') as zip_ref:
        zip_ref.extract(member=str(REL_MODEL_PATH), path=GENERATED_PATH)

    print(f'Model extracted to {MODEL_PATH}')
    print('Loading model...')

    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()

    print('Model loaded successfully')

    # TODO: get rif of hardcode nn img size, e.g. save and load dict, torch can do it
    dataset = VideoTableRecognitionDataset(
        input_video_path=input_video_path,
        nn_img_size=np.array([224, 224]),
        transforms=compose([pre_transforms(), post_transforms()])
    )
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    table_recognizer = NNRegressionBasedTableRecognizer(model=model, loader=dataloader)
    print('Finding frame by frame table polygon...')
    buffered_find_table_layout(layout_path, table_recognizer, get_video_resolution(input_video_path))
    print('Drawing table...')
    highlight_table(input_video_path, layout_path, video_with_table_path)
    print(f'Video with highlighted table saved in: {os.path.abspath(video_with_table_path)}')


if __name__ == '__main__':
    if os.path.exists(GENERATED_PATH):
        shutil.rmtree(GENERATED_PATH)
    os.makedirs(GENERATED_PATH, exist_ok=False)

    # print('Executing local data demo...')
    # draw_table(str(LOCAL_INPUT_VIDEO_PATH), str(LOCAL_LAYOUT_PATH), str(LOCAL_VIDEO_WITH_TABLE_PATH))

    print('Executing youtube data demo...')
    if os.path.exists(YOUTUBE_INPUT_VIDEO_PATH):
        os.remove(YOUTUBE_INPUT_VIDEO_PATH)
    download_youtube_video(str(YOTUBE_DATA_URL), str(DEMO_ROOT / 'tmp'))
    print('Video downloaded')
    for f in os.listdir(DEMO_ROOT / 'tmp'):
        path = DEMO_ROOT / 'tmp' / Path(f)
        cut_video(
            str(path), str(YOUTUBE_INPUT_VIDEO_PATH),
            parse_seconds_from_hh_mm_ss('00:00:00'),
            parse_seconds_from_hh_mm_ss('00:00:57')
            # parse_seconds_from_hh_mm_ss('00:07:40'),
            # parse_seconds_from_hh_mm_ss('00:08:00')
        )
        shutil.rmtree(DEMO_ROOT / 'tmp')
    print('Video cutted')
    draw_table(str(YOUTUBE_INPUT_VIDEO_PATH), str(YOUTUBE_LAYOUT_PATH), str(YOUTUBE_VIDEO_WITH_TABLE_PATH))
