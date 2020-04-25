import os
from table_recognition.find_table_polygon import table_polygon_layout
from table_recognition.highlight_table import highlight_table_video

if __name__ == '__main__':
    # Demo for the table rectangle recognition

    video_path = "../../data/sync/table_recognition_demo/table_recognition_demo_video.mp4"

    output_dir = "../../data/local/table_recognition_demo/"
    if os.path.exists(output_dir):
        print("Error! Output directory exists: ", output_dir)
        exit(1)
    else:
        os.makedirs(output_dir)

    layout_path = os.path.join(output_dir, "layout.txt")
    video_table_path = os.path.join(output_dir, "video.mp4")

    print("Finding frame by frame table polygon...")
    table_polygon_layout(video_path, layout_path)

    print("Drawing table...")
    highlight_table_video(video_path, layout_path, video_table_path)

    print("Video with highlighted table saved on:", video_table_path)
