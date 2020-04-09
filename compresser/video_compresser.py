import ffmpy
import argparse
import os


# Compresses video at input_filename using bitrate number (kbits/sec)
# and saves it to the output_filename.
# Deletes audio if delete_audio flag is set.
def compress(input_filename, output_filename, bitrate, delete_audio=True):
    inp_sz = os.path.getsize(input_filename)
    flags = '-y -strict -2 -b:v {}k'.format(bitrate)
    if delete_audio:
        flags += ' -an'
    inp = {input_filename: None}
    outp = {output_filename: flags}
    ff = ffmpy.FFmpeg(executable='C:/Users/NK/Downloads/ffmpeg-20200328-3362330-win64-static/ffmpeg-20200328-3362330-win64-static/bin/ffmpeg.exe', inputs=inp, outputs=outp)
    ff.run()
    out_sz = os.path.getsize(output_filename)
    print('Total {0:.3f}% lost'.format(100 * (inp_sz - out_sz) / inp_sz))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
		Compress input video with the given bitrate and save the result in output file.
		The estimated video size can be approximately computed as (bitrate) * (length in seconds) kB.
	''')
    parser.add_argument('-i', metavar='input', required=True,
                        help='input file name')
    parser.add_argument('-o', metavar='output', required=True,
                        help='output file name')
    parser.add_argument('-b', metavar='bitrate', type=int, default=400,
                        help='bitrate to compress input to (in kbits/sec)')
    args = parser.parse_args()
    compress(args.i, args.o, args.b)
