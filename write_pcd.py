from genericpath import isdir
import numpy as np
import open3d as o3d
import cv2
import os
import sys

from os.path import join
from apps.calibration.read_colmap import read_cameras_binary, read_points3d_binary
from easymocap.mytools.file_utils import read_json


def write_pcd(filename, points, colors):
    Vector3dVector = o3d.utility.Vector3dVector
    points = np.vstack(points)
    colors = np.vstack(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = Vector3dVector(points)
    if colors.dtype == np.uint8:
        colors = colors/255.
    pcd.colors = Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

def generate_color_bar(N):
    bar = ((np.arange(N)/(N-1))*255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    return colorbar

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='/home/shenwenhao/Downloads/fused.ply')
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument('--human', type=str, default='./startup/output-mono-smpl-robust/vertices/track/', help='path to human vertices json')
    parser.add_argument('--out', type=str, default='./sparse_vis/', help='path to store outputs')
    args = parser.parse_args()



    # Read background points
    points_suffix = os.path.basename(args.path).split('.')[-1]
    if points_suffix == "bin":
        points3D = read_points3d_binary(join(args.path, "points3D.bin"))
        keys = list(points3D.keys())
        xyz = np.stack([points3D[k].xyz for k in keys]) * args.scale
        rgb = np.stack([points3D[k].rgb for k in keys])
        key = 'sparse'
    elif points_suffix == "ply":
        points3D = o3d.io.read_point_cloud(args.path)
        key = 'dense'
        xyz = np.asarray(points3D.points) * args.scale
        rgb = np.asarray(points3D.colors) * 255.


    if os.path.isdir(args.human):
        humanid = f'assemble_{key}'
    else:
        humanid = os.path.basename(args.human).split('.')[0]
    verts = []
    for human_json in sorted(os.listdir(args.human)):
        print(join(args.human, human_json))
        human = read_json(join(args.human, human_json))
        vert = np.array(human[0]['vertices'])
        verts.append(vert)

    vert = np.concatenate(verts, axis=0)
    print(vert.shape)

    xyz = np.concatenate((xyz, vert), axis=0)
    rgb = np.concatenate((rgb, np.full(vert.shape, 255)), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb/255.)
    # o3d.visualization.draw_geometries([pcd])

    if not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)
    pcdname = join(args.out, f'sparse_{humanid}.ply')
    o3d.io.write_point_cloud(pcdname, pcd)
    scene_pcd = o3d.geometry.PointCloud()
