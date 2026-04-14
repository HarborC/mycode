import sys
sys.path.insert(0, '.')

from datasets.base.base_dataset import BaseDataset
from datasets.base.types import UnifiedClip
import os
import numpy as np
import os.path as osp
from pathlib import Path
from PIL import Image
from datasets.base.transforms import *


class KubricDataset(BaseDataset):
    STRIDE_CANDIDATES = [1, 2, 4]
    STRIDE_WEIGHTS = [0.45, 0.35, 0.2]

    def __init__(
        self,
        data_root=None,
        verbose=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'Kubric'
        self.data_root = Path(data_root)

        cache_path = f'data/dataset_cache/kubric_{self.mode}_cache.npy'
        os.makedirs('data/dataset_cache', exist_ok=True)
        if not os.path.exists(cache_path):
            self.sequences = []
            for p in sorted(self.data_root.iterdir()):
                if not p.is_dir():
                    continue
                seq = p.name
                seq_dir = p / seq if (p / seq).is_dir() else p
                if (
                    (seq_dir / f"{seq}.npy").exists()
                    and (seq_dir / f"{seq}_with_rank.npz").exists()
                    and (seq_dir / "frames").exists()
                ):
                    self.sequences.append(seq)
            if not self.sequences:
                raise RuntimeError(f"No valid Kubric scenes found under: {self.data_root}")
            np.save(cache_path, dict(sequences=self.sequences))
        else:
            npy = np.load(cache_path, allow_pickle=True).item()
            self.sequences = npy['sequences']

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} sequences in {data_root}', flush=True)

    def __len__(self):
        return len(self.sequences)

    def _get_clip(self, idx, resolution, rng):
        seq = self.sequences[idx]
        scene_dir = self.data_root / seq
        if (scene_dir / seq).is_dir():
            scene_dir = scene_dir / seq

        # ---- Load camera params ----
        rank = np.load(scene_dir / f"{seq}_with_rank.npz", allow_pickle=True)
        K_shared = np.asarray(rank["shared_intrinsics"], dtype=np.float32)        # [3,3]
        extrinsics_t34 = np.asarray(rank["extrinsics"], dtype=np.float32)         # [T_total,3,4], w2c

        frame_files = sorted((scene_dir / "frames").glob("*.png"))
        depth_files = sorted((scene_dir / "depths").glob("*.npy"))
        T_total = len(frame_files)
        frame_idxs = self._sample_frame_indices(T_total, rng)

        # ---- Load tracks and visibility ----
        ann = np.load(scene_dir / f"{seq}.npy", allow_pickle=True).item()
        coords_nt2 = np.asarray(ann["coords"], dtype=np.float32)                # [N, T_total, 2]
        visibility_nt = np.asarray(ann["visibility"], dtype=bool)               # [N, T_total]

        trajs_2d = np.transpose(coords_nt2[:, frame_idxs, :], (1, 0, 2))        # [T, N, 2]
        visibility = np.transpose(visibility_nt[:, frame_idxs], (1, 0))         # [T, N]

        # Sample depth at track locations from dense depth maps
        coords_depth = self._sample_depth_at_tracks(
            coords_nt2, np.asarray(ann["depth"], dtype=np.float32), frame_idxs) # [T, N]

        # ---- Compute initial valids before crop/resize ----
        valids = (
            np.isfinite(trajs_2d[..., 0])
            & np.isfinite(trajs_2d[..., 1])
            & np.isfinite(coords_depth)
            & (coords_depth > 0)
        )

        images, depths, poses, intrinsics = [], [], [], []
        pts3d_list, valid_mask_list = [], []
        instances = []

        clip_label = seq

        for t_i, fi in enumerate(frame_idxs):
            rgb_image = np.asarray(Image.open(frame_files[int(fi)]).convert("RGB"))
            depthmap = np.load(depth_files[int(fi)]).astype(np.float32).squeeze()

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :4] = extrinsics_t34[int(fi)]
            camera_pose = np.linalg.inv(w2c)

            frame_trajs = trajs_2d[t_i].copy().astype(np.float32)
            frame_valids = valids[t_i].copy()

            rgb_image, depthmap, K, _, _, frame_trajs, frame_valids = self._crop_resize_if_necessary(
                rgb_image, depthmap, K_shared.copy(), resolution,
                rng=rng, info=str(frame_files[int(fi)]),
                trajs_2d=frame_trajs, valids=frame_valids)

            pts3d_i, valid_mask_i, depthmap = self._process_depth(
                depthmap, K, camera_pose,
                label=f'{self.dataset_label}/{clip_label}', frame_id=str(int(fi)))

            images.append(self.transform(rgb_image))
            depths.append(depthmap.astype(np.float32))
            poses.append(camera_pose.astype(np.float32))
            intrinsics.append(K)
            instances.append(frame_files[int(fi)].name)
            pts3d_list.append(pts3d_i)
            valid_mask_list.append(valid_mask_i)

            trajs_2d[t_i] = frame_trajs
            valids[t_i] = frame_valids

        camera_poses = np.stack(poses, axis=0)
        intrinsics_arr = np.stack(intrinsics, axis=0)

        # ---- Build extrinsics [T, 4, 4] w2c ----
        extrinsics_w2c = np.zeros((len(frame_idxs), 4, 4), dtype=np.float32)
        extrinsics_w2c[:, :3, :4] = extrinsics_t34[frame_idxs]
        extrinsics_w2c[:, 3, 3] = 1.0

        # ---- Backproject tracks to world coords (using cropped intrinsics) ----
        trajs_3d_world = self._backproject_tracks_to_world(
            trajs_2d=trajs_2d,
            coords_depth=coords_depth,
            intrinsics=intrinsics_arr,
            extrinsics_w2c=extrinsics_w2c,
        )

        clip = UnifiedClip(
            images=torch.stack(images, dim=0),
            depths=np.stack(depths, axis=0),
            camera_poses=camera_poses,
            intrinsics=intrinsics_arr,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            visibility=visibility,
            valids=valids,
            dataset=self.dataset_label,
            label=clip_label,
            instances=instances,
            metadata={
                'has_tracks': True,
                'has_visibility': True,
                'has_trajs_3d_world': True,
            },
        )
        clip.pts3d = np.stack(pts3d_list, axis=0)
        clip.valid_mask = np.stack(valid_mask_list, axis=0)
        return clip

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_depth_at_tracks(
        coords_nt2: np.ndarray,
        dense_depth: np.ndarray,
        frame_idxs: np.ndarray,
    ) -> np.ndarray:
        """Sample dense depth maps at track 2D coordinates.

        Args:
            coords_nt2: [N, T_total, 2] float32 pixel coordinates.
            dense_depth: [T_total, H, W, 1] float32 depth maps.
            frame_idxs: array of selected frame indices.

        Returns:
            coords_depth: [len(frame_idxs), N] float32 depth at each track.
        """
        N, T_total, _ = coords_nt2.shape
        H, W = dense_depth.shape[1], dense_depth.shape[2]
        T = len(frame_idxs)
        coords_depth = np.full((T, N), np.nan, dtype=np.float32)

        for t_idx, fi in enumerate(frame_idxs):
            depth_map = dense_depth[int(fi), :, :, 0]  # [H, W]
            xy = coords_nt2[:, int(fi), :]             # [N, 2]

            # Round to nearest pixel
            ix = np.clip(np.round(xy[:, 0]).astype(np.int32), 0, W - 1)
            iy = np.clip(np.round(xy[:, 1]).astype(np.int32), 0, H - 1)

            vals = depth_map[iy, ix]
            coords_depth[t_idx] = vals

        return coords_depth

    @staticmethod
    def _backproject_tracks_to_world(
        trajs_2d: np.ndarray,
        coords_depth: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics_w2c: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct world-space track points from 2D coords + depth + K + w2c.

        Args:
            trajs_2d: [T, N, 2] float32 pixel coords.
            coords_depth: [T, N] float32 depth (ray distance).
            intrinsics: [T, 3, 3] pinhole intrinsics.
            extrinsics_w2c: [T, 4, 4] world-to-camera.

        Returns:
            trajs_3d_world: [T, N, 3] float32.
        """
        T, N, _ = trajs_2d.shape
        trajs_3d_world = np.full((T, N, 3), np.nan, dtype=np.float32)

        for t in range(T):
            uv = trajs_2d[t]            # [N, 2]
            z = coords_depth[t]         # [N]
            K = intrinsics[t]           # [3, 3]
            w2c = extrinsics_w2c[t]     # [4, 4]

            valid = (
                np.isfinite(uv[:, 0])
                & np.isfinite(uv[:, 1])
                & np.isfinite(z)
                & (z > 0)
            )
            if not np.any(valid):
                continue

            uv_valid = uv[valid]
            z_valid = z[valid]

            ones = np.ones((uv_valid.shape[0], 1), dtype=np.float32)
            pix = np.concatenate([uv_valid, ones], axis=-1)  # [M, 3]

            K_inv = np.linalg.inv(K).astype(np.float32)
            rays = (K_inv @ pix.T).T                          # [M, 3]
            ray_len = np.linalg.norm(rays, axis=1)
            z_cam = z_valid / ray_len
            pts_cam = rays * z_cam[:, None]

            pts_cam_h = np.concatenate(
                [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)],
                axis=-1,
            )

            c2w = np.linalg.inv(w2c).astype(np.float32)
            pts_world_h = (c2w @ pts_cam_h.T).T
            trajs_3d_world[t, valid] = pts_world_h[:, :3]

        return trajs_3d_world


if __name__ == '__main__':
    from datasets.utils.viser import visualize_dataset

    dataset = KubricDataset(
        data_root='/data2/d4rt/datasets/kubric',
        frame_num=48,
        resolution=[(512, 384)],
        mode='train',
        seed=42,
        sampling_mode="stride"
    )

    dataset[0].save_as_rrd('Kubric_clip.rrd')
    visualize_dataset(dataset, start_idx=0)
