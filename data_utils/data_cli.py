from argparse import ArgumentParser
from data_utils.utils import download_youtube_video, extract_images_from_video, frame_by_frame_play, cut_video
import dateparser
from datetime import timedelta


# def add_downloading_subparser(subparsers):
#     def run_downloading(args):
#         download_youtube_video(args.url, out_dir_path=args.out_dir_path, fix_name=True)
#
#     downloading_parser = subparsers.add_parser('download', help='Download video from YouTube')
#     downloading_parser.add_argument('url')
#     downloading_parser.add_argument('out_dir_path', default='.')
#     downloading_parser.set_defaults(func=run_downloading)


def add_cut_subparser(subparsers):
    def run_cut(args):
        def get_seconds(t):
            d = dateparser.parse(t)
            return timedelta(hours=d.hour, minutes=d.minute, seconds=d.second).total_seconds()

        from_s = get_seconds(args.left)
        to_s = get_seconds(args.right)
        cut_video(args.video_path, args.out_video_path, from_s, to_s)

    cut_parser = subparsers.add_parser('cut', help='Cut the video by time interval')
    cut_parser.add_argument('video_path', help='Path to the video')
    cut_parser.add_argument('out_video_path', help='Path to the output video')
    cut_parser.add_argument('left', help='The left bound of the time interval in hh:mm:ss format')
    cut_parser.add_argument('right', help='The right bound of the time interval in hh:mm:ss format')
    cut_parser.set_defaults(func=run_cut)


def add_extract_subparser(subparsers):
    def run_extraction(args):
        extract_images_from_video(args.video_path, args.out_path, args.step, args.skip, args.max_count)

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


def add_extract_manual_subparser(subparsers):
    def run_manual_extraction(args):
        frame_by_frame_play(args.video_path, skip_seconds=args.skip, frame_output_path=args.output,
                            frame_save_modifier=args.name_mixture)

    extract_manual_parser = subparsers.add_parser('extract_manual', help='Extract images from video manually')
    extract_manual_parser.add_argument('video_path', help='Path to the video')
    extract_manual_parser.add_argument('--skip', type=float, default=0,
                                       help='Time to skip in the beginning of the video in seconds')
    extract_manual_parser.add_argument('--name_mixture', default='',
                                       help='Text to add to file names')
    extract_manual_parser.add_argument('--output', default=None,
                                       help='Output directory. Directory of video is used by default.')
    extract_manual_parser.set_defaults(func=run_manual_extraction)


if __name__ == '__main__':
    argument_parser = ArgumentParser()
    subparsers = argument_parser.add_subparsers(dest='command')
    subparsers.required = True

    # add_downloading_subparser(subparsers)
    add_extract_subparser(subparsers)
    add_extract_manual_subparser(subparsers)
    add_cut_subparser(subparsers)

    arguments = argument_parser.parse_args()
    arguments.func(arguments)
