#!/usr/bin/env python3
# --------------------------------------------------------
# Standalone extract_frames for Waymo Open Dataset
# Only extracts raw frames (RGB, pose, LiDAR projections, calibration)
# from .tfrecord files, without cropping/resizing.
#
# Usage:
#   1) install: pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4
#   2) run: python preprocess_waymo.py --waymo_dir /path/to/waymo_dir --output_dir /path/to/output
# --------------------------------------------------------
import sys
import os
import os.path as osp
import json
from tqdm import tqdm
import PIL.Image
import numpy as np

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from concurrent.futures import ProcessPoolExecutor


def get_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Extract frames from Waymo tfrecord files')
    parser.add_argument('--waymo_dir', required=True, help='Directory containing .tfrecord files')
    parser.add_argument('--output_dir', default='data/waymo_raw', help='Output directory for extracted frames')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers')
    return parser


def _list_sequences(db_root):
    """Recursively find all .tfrecord files, returning (relative_dir, filename) pairs."""
    print('>> Looking for sequences in', db_root)
    res = []
    for dirpath, _, filenames in os.walk(db_root):
        for f in sorted(filenames):
            if f.endswith('.tfrecord'):
                rel_dir = osp.relpath(dirpath, db_root)
                res.append((rel_dir, f))
    print(f'    found {len(res)} sequences')
    return res


def extract_frames_one_seq(filename):
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils

    print('>> Opening', filename)
    dataset = tf.data.TFRecordDataset(filename, compression_type='')

    calib = None
    frames = []

    for data in tqdm(dataset, leave=False):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(data.numpy()))

        content = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _, range_image_top_pose = content

        views = {}
        frames.append((frame.context.name, views))

        # once in a sequence, read camera calibration info
        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append((cam.name,
                              dict(width=cam.width,
                                   height=cam.height,
                                   intrinsics=list(cam.intrinsic),
                                   extrinsics=list(cam.extrinsic.transform))))

        # convert LIDAR to pointcloud
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        # The distance between lidar points and vehicle frame origin.
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        for i, image in enumerate(frame.images):
            # select relevant 3D points for this view
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            cp_points_msk_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)

            pose = np.asarray(image.pose.transform).reshape(4, 4)
            timestamp = image.pose_timestamp

            rgb = tf.image.decode_jpeg(image.image).numpy()

            pix = cp_points_msk_tensor[..., 1:3].numpy().round().astype(np.int16)
            pts3d = points_all[mask.numpy()]

            views[image.name] = dict(img=rgb, pose=pose, pixels=pix, pts3d=pts3d, timestamp=timestamp)

    return calib, frames


def process_one_seq(db_root, output_dir, seq):
    out_dir = osp.join(output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    calib_path = osp.join(out_dir, 'calib.json')
    if osp.isfile(calib_path):
        print(f'   Skipping {seq} (already processed)')
        return

    try:
        with tf.device('/CPU:0'):
            calib, frames = extract_frames_one_seq(osp.join(db_root, seq))
    except RuntimeError:
        print(f'/!\\ Error with sequence {seq} /!\\', file=sys.stderr)
        return  # nothing is saved

    for f, (frame_name, views) in enumerate(tqdm(frames, leave=False)):
        for cam_idx, view in views.items():
            img = PIL.Image.fromarray(view.pop('img'))
            img.save(osp.join(out_dir, f'{f:05d}_{cam_idx}.jpg'))
            np.savez(osp.join(out_dir, f'{f:05d}_{cam_idx}.npz'), **view)

    with open(calib_path, 'w') as f:
        json.dump(calib, f)

    print(f'   Saved {len(frames)} frames from {seq}')


def main(waymo_root, output_dir, workers=8):
    sequences = _list_sequences(waymo_root)
    print(f'>> Extracting frames to {output_dir}')

    if workers == 1:
        for rel_dir, seq in sequences:
            process_one_seq(osp.join(waymo_root, rel_dir), osp.join(output_dir, rel_dir), seq)
    else:
        from multiprocessing import Pool
        args = [(osp.join(waymo_root, rel_dir), osp.join(output_dir, rel_dir), seq) for rel_dir, seq in sequences]
        with Pool(processes=workers) as pool:
            pool.starmap(process_one_seq, args)

    print(f'>> Done! All frames extracted to {output_dir}')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.waymo_dir, args.output_dir, workers=args.workers)
