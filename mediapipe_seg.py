import cv2
import mediapipe as mp
import os
from termcolor import colored
from os.path import join
import numpy as np
from glob import glob
import logging


def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def mywarn(text):
    myprint(text, 'warn')

def myerror(text):
    myprint(text, 'error')

def run_cmd(cmd, verbo=True):
    if verbo: 
        logging.warning('[run] ' + cmd)
    os.system(cmd)
    return []

mp_pose = mp.solutions.pose

def mp_seg(root_path, sub):
    image_dir = join(path, 'images', sub)
    image_files = sorted( glob(join(image_dir, '*.jpg')) )
    assert len(image_files) > 0, "Empty image directory!"
    print(f'Pose segmentation of {image_dir}:')
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5, 
        model_complexity=2, enable_segmentation=True) as pose:

        for idx, file in enumerate(image_files):
            image_name = os.path.basename(file).split('.')[0]
            image = cv2.imread(file)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #print(results.pose_landmarks)
            # Draw pose segmentation.
            segm_2class = results.segmentation_mask
            outdir = join(root_path, 'mask-mediapipe', sub)
            if segm_2class is None:
                logging.warning(f"This video:{image_dir} cannot be segmented!")
                if os.path.exists(outdir):
                    cmd = f'rm -r {outdir}'
                    run_cmd(cmd, verbo=True)
                return sub
            segm_mask = (segm_2class * 255).astype(np.uint8)
            os.makedirs(outdir, exist_ok=True)
            cv2.imwrite(join(outdir, image_name+'.png'), segm_mask)
            if args.debug:
                annotated_image = image.copy()
                red_img = np.zeros_like(annotated_image)
                annotated_image = annotated_image * segm_2class + red_img * (1 - segm_2class)
                cv2.imwrite("debug.png", annotated_image)
                segm_2class = cv2.imread("debug.png", 0)
                annotated_image[segm_2class==0] = 0
                cv2.imwrite("masked_img.png", annotated_image)
                os._exit(0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, nargs='+')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    for path in args.path:
        #false_imgdir_list = []
        false_imgdir_path = join(path, 'mp_error_list.txt')
        if os.path.exists(false_imgdir_path):
            backup_txt = join(path,'error_list_backup.txt')
            cmd = f'cp {false_imgdir_path} {backup_txt}'
            logging.warning("fault list already exists! Backup and Remove fault list")
            run_cmd(cmd, verbo=True)
            cmd = f'rm {false_imgdir_path}'
            run_cmd(cmd, verbo=True)
        if not os.path.isdir(path) or \
            not os.path.exists(join(path, 'images')):
            logging.error('{} not exist!'.format(path))
            continue
        for sub in sorted(os.listdir(join(path, 'images'))):
            false_imgdir =  mp_seg(path, sub)
            #print(false_imgdir)
            if false_imgdir is not None:
                with open(false_imgdir_path, 'a') as fp:
                    fp.write(false_imgdir+'\n')
