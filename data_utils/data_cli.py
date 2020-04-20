from argparse import ArgumentParser
from data_utils.utils import download_youtube_video, extract_images_from_video, frame_by_frame_play


# def run_downloading(args):
#     download_youtube_video(args.url, out_dir_path=args.out_dir_path, fix_name=True)


def run_extraction(args):
    extract_images_from_video(args.video_path, args.out_path, args.step, args.skip, args.max_count)


def run_manual_extraction(args):
    frame_by_frame_play(args.video_path, skip_seconds=args.skip, frame_output_path=args.output,
                        frame_save_modifier=args.name_mixture)


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    subparsers = argument_parser.add_subparsers(dest='command')
    subparsers.required = True

    # downloading_parser = subparsers.add_parser('download', help='Download video from YouTube')
    # downloading_parser.add_argument('url')
    # downloading_parser.add_argument('out_dir_path', default='.')
    # downloading_parser.set_defaults(func=run_downloading)

    extract_parser = subparsers.add_parser('extract', help='Extract images from a video by a fixed step')
    extract_parser.add_argument('video_path', help='Path to the video')
    extract_parser.add_argument('out_path', help='Path to the output directory')
    extract_parser.add_argument('--step', type=float, default=1,
                                help='Step between images to skip in seconds')
    extract_parser.add_argument('--skip', type=float, default=0,
                                help='Time to skip in the beginning of the video in seconds')
    extract_parser.add_argument('--max_count', type=int, default=10000,
                                help='Maximum number of images to extract')
    extract_parser.set_defaults(func=run_extraction)

    extract_manual_parser = subparsers.add_parser('extract_manual', help='Extract images from video manually')
    extract_manual_parser.add_argument('video_path', help='Path to the video')
    extract_manual_parser.add_argument('--skip', type=float, default=0,
                                       help='Time to skip in the beginning of the video in seconds')
    extract_manual_parser.add_argument('--name_mixture', default='',
                                       help='Text to add to file names')
    extract_manual_parser.add_argument('--output', default=None,
                                       help='Output directory. Directory of video is used by default.')
    extract_manual_parser.set_defaults(func=run_manual_extraction)

    arguments = argument_parser.parse_args()
    arguments.func(arguments)
