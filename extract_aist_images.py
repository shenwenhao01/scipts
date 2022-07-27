# extract image from videos
import os
from os.path import join
from glob import glob
import ffmpeg

extensions = ['.mp4','.MP4']

import ffmpeg
import numpy as np
import cv2


def ffmpeg_video_read(video_path, fps=None):
    """Video reader based on FFMPEG.
    This function supports setting fps for video reading. It is critical
    as AIST++ Dataset are constructed under exact 60 fps, while some of
    the AIST dance videos are not percisely 60 fps.
    Args:
        video_path: A video file.
        fps: Use specific fps for video reading. (optional)
    Returns:
        A `np.array` with the shape of [seq_len, height, width, 3]
    """
    assert os.path.exists(video_path), f'{video_path} does not exist!'
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    video_info = next(stream for stream in probe['streams']
                        if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    stream = ffmpeg.input(video_path)
    if fps:
        stream = ffmpeg.filter(stream, 'fps', fps=fps, round='down')
    stream = ffmpeg.output(stream, 'pipe:', format='rawvideo', pix_fmt='rgb24')
    out, _ = ffmpeg.run(stream, capture_stdout=True, quiet=True)
    out = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return out.copy()

def run(cmd):
    print(cmd)
    os.system(cmd)

def extract_images(path, ffmpeg, image):
    videos = sorted(sum([
        glob(join(path, 'shorts', '*'+ext)) for ext in extensions
        ], [])
    )
    #videos = videos[0:2450]
    #videos = videos[2450:4900]
    #videos = videos[4900:4900+2450]
    videos = videos[4900+2450:]
    print(len(videos))
    for videoname in videos:
        videoname = videoname.replace('shorts', args.video)
        assert videoname, f'{videoname} not exists!!!'
        sub = '.'.join(os.path.basename(videoname).split('.')[:-1])
        sub = sub.replace(args.strip, '')
        outpath = join(path, image, sub)
        if os.path.exists(outpath) and len(os.listdir(outpath)) > 10 and not args.restart:
            continue
        os.makedirs(outpath, exist_ok=True)
        video = ffmpeg_video_read(videoname, fps=args.fps)            # nframe x height x width x 3
        out = video[ 0 : -1 : args.interval]
        #print(out.shape)
        for i in range(out.shape[0]):
            #print(join(outpath, f"{i:06d}.jpg".format(i)))
            cv2.imwrite(join(outpath, f"{i:06d}.jpg".format(i)), out[i][...,::-1].astype(np.uint8))
        print(f"Finish Writing {outpath}")
        #utils.ffmpeg_video_write(out, outpath, fps=10)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--strip', type=str, default='')
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--video', type=str, default='videos')
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--transpose', type=int, default=-1)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--ffmpeg', type=str, default='ffmpeg')
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    extract_images(args.path, args.ffmpeg, args.image)
