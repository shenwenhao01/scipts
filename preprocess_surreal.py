'''
Python2
'''
import json
import os
import re
import sys
import scipy.io as sio
from os.path import join
import cv2
import argparse
import PIL.Image
import math
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import logging
import numpy as np
from tqdm import tqdm
import transforms3d

SMPL_PATH = '/nas/home/shenwenhao/SMPL/SMPL_python_v.1.0.0/'
sys.path.append(SMPL_PATH)
from smpl_webuser.serialization import load_model


def file_ext(name):
    return str(name).split('.')[-1]

def is_video_ext(fname):
    ext = file_ext(fname).lower()
    return '{}'.format(ext) in ['mp4'] # type: ignore

def read_json(path):
    assert os.path.exists(path), path
    with open(path) as f:
        try:
            data = json.load(f)
        except:
            print('Reading error {}'.format(path))
            data = []
    return data

def save_json(file, data):
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].astype(np.float32).tolist()
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

def get_frame(filevideo, t=0):
    cap = cv2.VideoCapture(filevideo)
    cap.set(propId=1, value=t)
    ret, frame = cap.read()
    #frame = frame[:, :, [2, 1, 0]]
    return frame

def get_mask(segmfile, t=0):
    masks = sio.loadmat(segmfile)
    mask = masks['segm_{}'.format(t+1)]
    return mask

def get_extrinsic(T):
    # Take the first 3 columns of the matrix_world in Blender and transpose.
    # This is hard-coded since all images in SURREAL use the same.
    R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
    # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
    #                               (0., -1, 0., -1.0),
    #                               (-1., 0., 0., 0.),
    #                               (0.0, 0.0, 0.0, 1.0)))

    # Convert camera location to translation vector used in coordinate changes
    T_world2bcam = -1 * np.dot(R_world2bcam, T)

    # Following is needed to convert Blender camera to computer vision camera
    R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
    T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

    # Put into 3x4 matrix
    RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
    return RT, R_world2cv, T_world2cv

def rotateBody(RzBody, pelvisRotVec):
    angle = np.linalg.norm(pelvisRotVec)
    Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
    globRotMat = np.dot(RzBody, Rpelvis)
    R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
    globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
    globRotVec = globRotAx * globRotAngle
    return globRotVec

def project(xyz, K, R, T):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, R.T) + T.T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_filelist(root_dir):
    Filelist = []
    for home, dirs, files in os.walk(root_dir):
        for f in files:
            filename = join(home, f)
            if is_video_ext(filename) and os.path.isfile(filename):
                Filelist.append(filename)
                # Filelist.append( filename)
    return Filelist

def main(source_dir, dest_dir):
    # Hard-core params
    H, W = 240, 320
    pad = 4
    out_res = 128
    half_pad = pad / 2
    t = 0
    intrinsic = np.array([ [600,   0, 160],
                           [  0, 600, 120],
                           [  0,   0,   1] ], dtype=np.float32)
    #input_videos = [str(f) for f in sorted(Path(source_dir).rglob('*')) if is_video_ext(f) and os.path.isfile(f)]
    input_videos = get_filelist(source_dir)
    input_videos = input_videos[3522:]
    print("Total number of videos: {}".format(len(input_videos)))
    #print(input_videos[0])
    for idx, videoname in enumerate(tqdm(input_videos)):
        segmname = videoname.replace('.mp4', '_segm.mat')
        mask = get_mask(segmname)
        infoname = videoname.replace('.mp4', '_info.mat')
        info = sio.loadmat(infoname)

        x, y, w, h = cv2.boundingRect(mask)
        if x < half_pad or y < half_pad or x+w > W-half_pad or y+h > H-half_pad:
            logging.warning("\nThis man may exceed the image range: {} !".format(videoname))
            continue
    
        # save masked img; set bkgd = Black
        try:
            rgb = get_frame(videoname, t)       # BGR
            rgb[mask==0] = 0
        except:
            logging.warning('\nSomething may be wrong with {}'.format(videoname))
            continue
        crop_size = np.max((w,h)) + pad
        center = (y+h/2, x+w/2)

        aa = center[0]-crop_size/2
        ab = center[0]+crop_size/2
        ba = center[1]-crop_size/2
        bb = center[1]+crop_size/2
        if aa < 0 or ab > H or ba < 0 or bb > W :
            logging.warning("\nSkip large body: {}".format(videoname))
            continue
        rgb = rgb[aa : ab, ba : bb]
        assert rgb.shape[0] == rgb.shape[1]
        rgb = cv2.resize(rgb, (out_res, out_res), interpolation=cv2.INTER_NEAREST)
        assert rgb.shape[0]==rgb.shape[1]==128

        # SMPL model
        zrot = info['zrot'][0][0]        # body rotation in euler angles
        RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                        (math.sin(zrot), math.cos(zrot), 0),
                        (0, 0, 1)))
        print(info['joints3D'].shape)
        if len(info['joints3D'].shape) != 3:
            logging.warning("\nSkip invalid joints3D data: {}".format(videoname))
            continue
        joints3D = info['joints3D'][:, :, t].T
        pose = info['pose'][:, t]
        pose[0:3] = rotateBody(RzBody, pose[0:3])
        
        # Set model shape
        # <========= LOAD SMPL MODEL BASED ON GENDER
        if info['gender'][0] == 0:  # f
            m = load_model(os.path.join(SMPL_PATH, 'models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'))
        elif info['gender'][0] == 1:  # m
            m = load_model(os.path.join(SMPL_PATH, 'models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'))
        # =========>
        root_pos = m.J_transformed.r[0]
        m.betas[:] = info['shape'][:, 0]
        # Set model pose
        m.pose[:] = pose
        # Set model translation
        m.trans[:] = joints3D[0] - root_pos
        smpl_vertices = m.r

        # Calculate K and RT
        RT, R, T = get_extrinsic(info['camLoc'])
        K = intrinsic.copy()
        K[0, 2] -= (center[1]-crop_size/2)
        K[1, 2] -= (center[0]-crop_size/2)
        K[:2] *= (128. / crop_size)

        if args.debug:
            print(K)
            proj_smpl_vertices = project(smpl_vertices, K, R, T)
            import matplotlib
            matplotlib.use('agg')
            import matplotlib.pyplot as plt
            plt.clf()
            #plt.subplot(1, 1, 1)
            plt.imshow(rgb)
            plt.scatter(proj_smpl_vertices[:, 0], proj_smpl_vertices[:, 1], 1)
            plt.savefig("debug.png")

        # Save
        ret = {}
        ret['K'] = K
        ret['RT'] = RT
        smpl_params = {
            "shape": info['shape'][:, 0],
            "pose" : pose[3:],                  # 69
            "Th"   : joints3D[0] - root_pos,
            "Rh"   : pose[:3],
            "gender": info['gender'][0].item(),        # 0:female ; 1:male
        }
        ret.update( smpl_params )
        ret['vertices'] = smpl_vertices
        sub = os.path.basename(videoname).split('.')[0]
        img_path = join(args.outdir, sub + '.jpg')
        cv2.imwrite(img_path, rgb)
        json_path = join(args.outdir, sub + '.json')
        save_json(file=json_path, data=ret)
        if args.debug:
            meta = read_json(json_path)
            meta = {key:np.array(meta[key], dtype=np.float32) for key in meta if isinstance(meta[key], list)}
            reproj_smpl_vertices = project(meta['vertices'], meta['K'], meta['RT'][:, :3], meta['RT'][:, 3])
            plt.clf()
            #plt.subplot(1, 1, 1)
            plt.imshow(rgb)
            plt.scatter(reproj_smpl_vertices[:, 0], reproj_smpl_vertices[:, 1], 1)
            plt.savefig("debug_reproj.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/nas/home/shenwenhao/surreal/data/cmu/val/run0/ung_138_35',
                        help='Path to the SURREAL file')
    parser.add_argument('--outdir', type=str, default='.',
                        help='Output root path')
    parser.add_argument('--debug', action='store_true',
                        help='Debug')
    args = parser.parse_args()
    main(source_dir=args.path, dest_dir=args.outdir)