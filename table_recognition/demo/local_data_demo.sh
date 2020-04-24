export PYTHONPATH=$PYTHONPATH:../..

VIDEO_PATH="../../data/sync/table_recognition_demo/table_recongition_demo_video.mp4"
LAYOUT_PATH="layout.txt"
VIDEO_WITH_TABLE_PATH=video.mp4

echo "Finding frame by frame table polygon..."
python3 ../find_table_polygon.py --video $VIDEO_PATH --layout $LAYOUT_PATH
cat $LAYOUT_PATH

echo "Drawing table..."
python3 ../highlight_table.py --

python3 ../highlight_table.py --video $VIDEO_PATH --layout $LAYOUT_PATH --output $VIDEO_WITH_TABLE_PATH

echo "Video with highlighted table saved on: $VIDEO_WITH_TABLE_PATH"
