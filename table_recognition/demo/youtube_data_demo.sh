export PYTHONPATH=$PYTHONPATH:../..

VIDEO_FOLDER="youtube_folder"
VIDEO_PATH="$VIDEO_FOLDER/youtube_video.mp4"
CUT_VIDEO_PATH="$VIDEO_FOLDER/cut_youtube_video.mp4"
LAYOUT_PATH="layout.txt"
VIDEO_WITH_TABLE_PATH=youtube_video_table.mp4


echo "Downloading video from youtube..."
python3 demo_download.py
mv "$VIDEO_FOLDER/$(ls $VIDEO_FOLDER)" "$VIDEO_PATH"
echo "YouTube vide was saved in $VIDEO_PATH"

python3 ../../data_utils/data_cli.py cut "$VIDEO_PATH" "$CUT_VIDEO_PATH" "00:00:40" "00:00:50"
mv "$CUT_VIDEO_PATH" "$VIDEO_PATH"

echo "Finding frame by frame table polygon..."
python3 ../find_table_polygon.py --video $VIDEO_PATH --layout $LAYOUT_PATH

echo "Drawing table..."
python3 ../highlight_table.py --video $VIDEO_PATH --layout $LAYOUT_PATH --output $VIDEO_WITH_TABLE_PATH

echo "Video with highlighted table was saved in: $VIDEO_WITH_TABLE_PATH"
