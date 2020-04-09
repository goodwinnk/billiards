import os
from compresser.video_compresser import compress

def test_compress():
    tests = [
        (
            '../video/compress_test.mp4',
            '../video/compress_test_output.mp4',
            50,
            2000,
        ),
    ]
    for inp_file, outp_file, bitrate, high_threshold_sz in tests:
        compress(inp_file, outp_file, bitrate)
        sz = os.path.getsize(outp_file)
        os.remove(outp_file)
        assert sz <= high_threshold_sz
