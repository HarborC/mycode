"""Visualize views from any dataset.

Usage:
    python -m datasets.utils.viser --help
    # Or directly in code:
    from datasets.utils.viser import visualize_dataset
    visualize_dataset(dataset, start_idx=0)
"""

import sys
import os
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import viser
import viser.transforms as tf


def depthmap_to_pointcloud(depthmap, intrinsics, camera_pose, rgb=None, valid_mask=None, downsample=4):
    """Convert depthmap + intrinsics + pose to point cloud."""
    if valid_mask is None:
        valid_mask = depthmap > 0

    H, W = depthmap.shape[:2]
    v, u = np.mgrid[0:H:downsample, 0:W:downsample]
    z = depthmap[::downsample, ::downsample]
    mask = valid_mask[::downsample, ::downsample] & (z > 0)

    u = u[mask].astype(np.float64)
    v = v[mask].astype(np.float64)
    z = z[mask].astype(np.float64)

    fx, fy = intrinsics[0, 0].astype(np.float64), intrinsics[1, 1].astype(np.float64)
    cx, cy = intrinsics[0, 2].astype(np.float64), intrinsics[1, 2].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)

    R = camera_pose[:3, :3].astype(np.float64)
    t = camera_pose[:3, 3].astype(np.float64)
    pts_world = (R @ pts_cam.T).T + t

    if rgb is not None:
        if rgb.ndim == 3 and rgb.shape[0] == 3:  # CHW -> HWC
            rgb = rgb.transpose(1, 2, 0)
        colors = rgb[::downsample, ::downsample][mask].astype(np.float32) / 255.0
    else:
        colors = np.ones((len(pts_world), 3), dtype=np.float32) * 0.8

    return pts_world.astype(np.float32), colors


def _to_numpy(v):
    if hasattr(v, 'cpu'):
        return v.cpu().numpy()
    return np.asarray(v)


def visualize_dataset(dataset, start_idx=0, share=False, point_size=0.01):
    """Visualize a dataset with on-demand loading. Use Prev/Next Sample buttons in GUI."""
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print(f"Dataset has {len(dataset)} samples. Starting at index {start_idx}.")

    all_point_nodes = []
    all_frustum_nodes = []
    all_frame_nodes = []
    all_info_text = []
    current_idx = [start_idx]
    first_frame_pose = [None]  # camera_pose of first frame, updated on each load
    initialized_clients = set()  # clients that have already had their camera set

    init_frame_num = getattr(dataset, 'frame_num', 48)

    with server.gui.add_folder("Controls"):
        gui_point_size = server.gui.add_slider("Point size", min=0.01, max=0.05, step=1e-3, initial_value=point_size)
        gui_frame_num = server.gui.add_number("Num frames", min=1, max=128, step=1, initial_value=init_frame_num)
        gui_sample_label = server.gui.add_markdown(f"**Sample**: {start_idx} / {len(dataset)-1}")
        gui_prev_sample = server.gui.add_button("◀ Prev Sample")
        gui_next_sample = server.gui.add_button("Next Sample ▶")
        with server.gui.add_folder("Frame Navigation"):
            gui_frame_idx = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0)
            gui_prev_frame = server.gui.add_button("◀ Prev Frame")
            gui_next_frame = server.gui.add_button("Next Frame ▶")
            gui_playing = server.gui.add_checkbox("Auto Play", False)
            gui_framerate = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=5)
        gui_show_all = server.gui.add_checkbox("Show all frames", True)
        gui_show_cameras = server.gui.add_checkbox("Show cameras", True)

    def _set_client_camera(client):
        """Set client camera to first frame's viewpoint (camera-to-world pose)."""
        pose = first_frame_pose[0]
        if pose is None:
            return
        cam_pos = pose[:3, 3]
        look_at = cam_pos + pose[:3, 2]   # +Z axis = look direction (OpenCV)
        up_dir = -pose[:3, 1]             # -Y axis = up direction (OpenCV)
        client.camera.position = cam_pos
        client.camera.look_at = look_at
        client.camera.up_direction = up_dir

    @server.on_client_connect
    def _(client):
        @client.camera.on_update
        def _(cam):
            # Only jump to first frame once per client connection
            if client.client_id not in initialized_clients:
                initialized_clients.add(client.client_id)
                _set_client_camera(client)

    def load_and_build(dataset_idx):
        nonlocal all_point_nodes, all_frustum_nodes, all_frame_nodes, all_info_text

        # clear old scene
        for n in all_point_nodes + all_frustum_nodes + all_frame_nodes:
            n.remove()
        for t in all_info_text:
            t.remove()
        all_point_nodes, all_frustum_nodes, all_frame_nodes, all_info_text = [], [], [], []

        # Update dataset frame_num if it has this attribute
        if hasattr(dataset, 'frame_num'):
            dataset.frame_num = gui_frame_num.value

        print(f"Loading sample {dataset_idx} with {gui_frame_num.value} frames...")
        clip = dataset[dataset_idx]
        T = clip.images.shape[0]

        gui_sample_label.content = f"**Sample**: {dataset_idx} / {len(dataset)-1}"
        gui_frame_idx.max = max(T - 1, 0)
        gui_frame_idx.value = 0

        meta_lines = []
        meta_lines.append(f"**dataset**: {clip.dataset}")
        meta_lines.append(f"**label**: {clip.label}")
        meta_lines.append(f"**frames**: {T}")
        info_text = server.gui.add_markdown(" | ".join(meta_lines))
        all_info_text.append(info_text)

        point_nodes, frustum_nodes, frame_nodes = [], [], []
        for i in range(T):
            rgb = _to_numpy(clip.images[i])
            if rgb.ndim == 3 and rgb.shape[0] == 3:  # CHW -> HWC
                rgb = rgb.transpose(1, 2, 0)
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.1 else rgb.astype(np.uint8)

            depthmap = np.asarray(clip.depths[i], dtype=np.float32)
            camera_pose = np.asarray(clip.camera_poses[i], dtype=np.float64)
            intrinsics = np.asarray(clip.intrinsics[i], dtype=np.float64)
            valid_mask = clip.valid_mask[i] if clip.valid_mask is not None else (depthmap > 0)
            valid_mask = np.asarray(valid_mask)

            pts, colors = depthmap_to_pointcloud(depthmap, intrinsics, camera_pose, rgb=rgb, valid_mask=valid_mask)
            if len(pts) == 0:
                continue

            frame_node = server.scene.add_frame(f"/s/f{i}", show_axes=False)
            frame_nodes.append(frame_node)

            pc_node = server.scene.add_point_cloud(
                name=f"/s/f{i}/pc", points=pts, colors=colors,
                point_size=gui_point_size.value, point_shape="rounded",
            )
            point_nodes.append(pc_node)

            H, W = depthmap.shape[:2]
            fov = 2 * np.arctan2(H / 2, intrinsics[0, 0])
            frustum_node = server.scene.add_camera_frustum(
                f"/s/f{i}/frustum", fov=fov, aspect=W/H, scale=0.15,
                image=rgb[::4, ::4],
                wxyz=tf.SO3.from_matrix(camera_pose[:3, :3]).wxyz,
                position=camera_pose[:3, 3],
            )
            frustum_node.visible = gui_show_cameras.value
            frustum_nodes.append(frustum_node)

            inst = clip.instances[i] if i < len(clip.instances) else ''
            label_text = f"{i} {inst}" if inst else str(i)
            server.scene.add_label(
                f"/s/f{i}/label", text=label_text,
                wxyz=tf.SO3.from_matrix(camera_pose[:3, :3]).wxyz,
                position=camera_pose[:3, 3],
            )

        all_point_nodes[:] = point_nodes
        all_frustum_nodes[:] = frustum_nodes
        all_frame_nodes[:] = frame_nodes

        # Fix slider max to match actual number of built frames (some may be skipped)
        gui_frame_idx.max = max(len(frame_nodes) - 1, 0)
        gui_frame_idx.value = 0

        # show all frames by default
        for i, fn in enumerate(all_frame_nodes):
            fn.visible = True

        # Store first frame pose and update camera for already-connected clients
        if T > 0:
            first_frame_pose[0] = np.asarray(clip.camera_poses[0], dtype=np.float64)
            for client in server.get_clients().values():
                _set_client_camera(client)

    load_and_build(current_idx[0])

    @gui_frame_num.on_update
    def _(_):
        load_and_build(current_idx[0])

    @gui_prev_sample.on_click
    def _(_):
        current_idx[0] = max(0, current_idx[0] - 1)
        load_and_build(current_idx[0])

    @gui_next_sample.on_click
    def _(_):
        current_idx[0] = min(len(dataset) - 1, current_idx[0] + 1)
        load_and_build(current_idx[0])

    @gui_prev_frame.on_click
    def _(_):
        n = gui_frame_idx.max + 1
        gui_frame_idx.value = (gui_frame_idx.value - 1) % n

    @gui_next_frame.on_click
    def _(_):
        n = gui_frame_idx.max + 1
        gui_frame_idx.value = (gui_frame_idx.value + 1) % n

    @gui_playing.on_update
    def _(_):
        if gui_playing.value and gui_show_all.value:
            gui_show_all.value = False  # auto-disable "show all" when playing
        gui_frame_idx.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value

    @gui_frame_idx.on_update
    def _(_):
        if not gui_show_all.value:
            for i, fn in enumerate(all_frame_nodes):
                fn.visible = (i == gui_frame_idx.value)

    @gui_show_all.on_update
    def _(_):
        gui_frame_idx.disabled = gui_show_all.value
        gui_prev_frame.disabled = gui_show_all.value
        gui_next_frame.disabled = gui_show_all.value
        for i, fn in enumerate(all_frame_nodes):
            fn.visible = gui_show_all.value or (i == gui_frame_idx.value)

    @gui_show_cameras.on_update
    def _(_):
        for fn in all_frustum_nodes:
            fn.visible = gui_show_cameras.value

    while True:
        if gui_playing.value and not gui_show_all.value:
            n = gui_frame_idx.max + 1
            gui_frame_idx.value = (gui_frame_idx.value + 1) % n
        for pc in all_point_nodes:
            pc.point_size = gui_point_size.value
        time.sleep(1.0 / gui_framerate.value)

