from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class UnifiedClip:
    """Clip-level data container for all datasets.

    Contains both raw loaded data and post-processed fields
    (pts3d, valid_mask, true_shape, etc.) computed by ``BaseDataset.__getitem__``.

    Fields
    ------
    images : ndarray
        ``[T, H, W, 3]`` uint8 RGB images (from _get_clip),
        or ``[T, 3, H, W]`` float32 tensor (after transform in __getitem__).
    depths : ndarray
        ``[T, H, W]`` float32 depth maps.
    camera_poses : ndarray
        ``[T, 4, 4]`` camera-to-world (c2w) poses.
    intrinsics : ndarray
        ``[T, 3, 3]`` pinhole intrinsics.

    trajs_2d : ndarray | None
        ``[T, N, 2]`` float32 2-D track pixel coordinates.
    trajs_3d_world : ndarray | None
        ``[T, N, 3]`` float32 3-D world-space track positions.
    visibility : ndarray | None
        ``[T, N]`` bool  track visibility mask.

    pts3d : ndarray | None
        ``[T, H, W, 3]`` float32 3-D point maps.
        Computed by ``BaseDataset.__getitem__`` from depth + intrinsics + pose.
    valid_mask : ndarray | None
        ``[T, H, W]`` bool valid-pixel masks.
    normal : ndarray | None
        ``[T, H, W, 3]`` float32 surface normals, or None.

    true_shape : ndarray
        ``[T, 2]`` int32 original (height, width) before crop/resize.
    idx : tuple
        ``(dataset_idx, ar_idx, view_idx)`` sample index.
    z_far : float
        Far clipping distance for depth filtering.

    dataset : str
    label : str
    instances : list[str]
    metadata : dict
    """

    images: np.ndarray             # [T, H, W, 3] uint8 or [T, 3, H, W] float32 tensor
    depths: np.ndarray             # [T, H, W] float32
    camera_poses: np.ndarray       # [T, 4, 4] c2w
    intrinsics: np.ndarray         # [T, 3, 3]

    trajs_2d: Optional[np.ndarray] = None
    trajs_3d_world: Optional[np.ndarray] = None
    visibility: Optional[np.ndarray] = None

    pts3d: Optional[np.ndarray] = None      # [T, H, W, 3] float32
    valid_mask: Optional[np.ndarray] = None  # [T, H, W] bool
    normal: Optional[np.ndarray] = None      # [T, H, W, 3] float32

    true_shape: Optional[np.ndarray] = None   # [T, 2]
    idx: Optional[tuple] = None
    z_far: float = 0.0

    dataset: str = ""
    label: str = ""
    instances: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_frame_image(self, t: int) -> np.ndarray:
        """Get frame t as [H, W, 3] uint8 numpy array."""
        img = self.images[t]
        if img.ndim == 3 and img.shape[0] == 3:  # [3, H, W] tensor
            # denormalize from [-1, 1] or [0, 1]
            img = img.transpose(1, 2, 0)
            if img.max() <= 1.5:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def _project_world_to_image(pts_world, K, E):
        """Project world points to image. pts_world: [N,3], K: [3,3], E: [4,4] w2c -> [N,2]."""
        N = pts_world.shape[0]
        h = np.concatenate([pts_world, np.ones((N, 1), dtype=np.float32)], axis=1)
        cam = (E @ h.T).T[:, :3]
        z = cam[:, 2:3]
        valid = (z[:, 0] > 1e-3) & np.isfinite(cam).all(1)
        uv = np.full((N, 2), np.nan, dtype=np.float32)
        if valid.any():
            uv[valid, 0] = cam[valid, 0] / z[valid, 0] * K[0, 0] + K[0, 2]
            uv[valid, 1] = cam[valid, 1] / z[valid, 0] * K[1, 1] + K[1, 2]
        return uv, valid

    # ------------------------------------------------------------------
    # Rerun export
    # ------------------------------------------------------------------

    def save_as_rrd(self, path: str):
        """Log clip to a rerun .rrd file for offline inspection.

        Adapted from old_code/verify_datasets.py ``log_clip_to_rerun``.

        Renders a Blueprint with:
        - Top row: RGB + reprojection view, optional depth view, optional normal view
        - Bottom row: 3D scene, info panel

        Args:
            path: output ``.rrd`` file path.
        """
        import rerun as rr
        import rerun.blueprint as rrb
        from scipy.spatial.transform import Rotation

        rrd_path = Path(path)
        rrd_path.parent.mkdir(parents=True, exist_ok=True)

        T = self.images.shape[0]
        H, W = int(self.true_shape[0, 0]), int(self.true_shape[0, 1])

        has_depth   = self.depths is not None
        has_normals = self.normal is not None
        has_trajs3d = self.trajs_3d_world is not None

        # ── Blueprint ─────────────────────────────────────────────────
        top_views = [
            rrb.Spatial2DView(
                name="RGB & Reprojection",
                origin="world/camera/image",
                contents=["+ $origin/**"],
                background=[30, 30, 30],
            ),
        ]
        if has_depth:
            top_views.append(rrb.Spatial2DView(
                name="Depth",
                origin="world/camera/image",
                contents=["+ world/camera/image/depth"],
                background=[30, 30, 30],
            ))
        if has_normals:
            top_views.append(rrb.Spatial2DView(
                name="Normals",
                origin="vis/normals",
                background=[30, 30, 30],
            ))

        view_3d = rrb.Spatial3DView(
            name="3D Scene",
            origin="/",
            contents=["+ world/**"],
            background=[20, 20, 20],
        )

        info_view = rrb.TextDocumentView(name="Info", origin="info")

        layout = rrb.Vertical(
            rrb.Horizontal(*top_views),
            rrb.Horizontal(view_3d, info_view, column_shares=[4, 1]),
            row_shares=[1, 2],
        )
        blueprint = rrb.Blueprint(layout, collapse_panels=True)

        # ── Recording ─────────────────────────────────────────────────
        app_id = f"{self.dataset}/{self.label}" if self.dataset else "UnifiedClip"
        with rr.RecordingStream(application_id=app_id, make_default=False) as rec:
            rec.save(str(rrd_path), default_blueprint=blueprint)

            # ── Metadata ──────────────────────────────────────────────
            info_lines = [
                f"# {self.dataset} / {self.label}",
                f"- **Frames**: {T}",
                f"- **Resolution**: {W} x {H}",
                f"- **Depth**: {'yes' if has_depth else 'no'}",
                f"- **Normals**: {'yes' if has_normals else 'no'}",
                f"- **2D tracks**: {'yes' if self.trajs_2d is not None else 'no'}",
                f"- **3D tracks**: {'yes' if has_trajs3d else 'no'}",
                f"- **z_far**: {self.z_far}",
                "\n### Visualization",
                "- **Green points**: Ground truth 2D trajectories",
                "- **Red points**: Reprojected 3D trajectories",
                "- **Orange points**: 3D points backprojected from depth map",
            ]
            rec.log("info", rr.TextDocument("\n".join(info_lines), media_type=rr.MediaType.MARKDOWN))

            # ── Estimate frustum scale ────────────────────────────────
            frustum_scale = 0.3
            if has_trajs3d:
                scene_extent = float(np.linalg.norm(
                    self.trajs_3d_world[0].max(0) - self.trajs_3d_world[0].min(0)))
                frustum_scale = max(0.05, scene_extent * 0.05)
            elif has_depth:
                median_depths = [
                    float(np.median(d[d > 0]))
                    for d in self.depths if np.any(d > 0)
                ]
                if median_depths:
                    frustum_scale = float(np.median(median_depths)) * 0.1

            # ── Per-frame data ────────────────────────────────────────
            for t in range(T):
                rec.set_time("frame", sequence=t)

                K = self.intrinsics[t]
                c2w = self.camera_poses[t]
                w2c = np.linalg.inv(c2w)

                # Camera transform & pinhole
                quat = Rotation.from_matrix(c2w[:3, :3]).as_quat()  # [x,y,z,w]
                rec.log("world/camera", rr.Transform3D(
                    translation=c2w[:3, 3],
                    quaternion=rr.Quaternion(xyzw=quat),
                ))
                rec.log("world/camera/image", rr.Pinhole(
                    image_from_camera=K, width=W, height=H,
                    image_plane_distance=frustum_scale,
                ))

                # RGB image
                rec.log("world/camera/image", rr.Image(self._get_frame_image(t)))

                # Depth
                if has_depth:
                    rec.log("world/camera/image/depth", rr.DepthImage(
                        self.depths[t], meter=1.0, colormap="Turbo",
                        point_fill_ratio=1.0,
                    ))

                # Normals
                if has_normals and self.normal[t] is not None:
                    normal_vis = ((self.normal[t] * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
                    rec.log("vis/normals", rr.Image(normal_vis))

                # Tracks: 2D GT vs reprojection, 3D points
                if has_trajs3d and self.trajs_2d is not None and self.visibility is not None:
                    valid = valid & self.visibility[t].astype(bool)
                    if valid.any():
                        pts2d_gt = self.trajs_2d[t][valid].astype(np.float32)
                        pts3d_v = self.trajs_3d_world[t][valid].astype(np.float32)
                        uv_reproj, reproj_valid = self._project_world_to_image(
                            pts3d_v, K.astype(np.float32), w2c.astype(np.float32))

                        rec.log("world/camera/image/gt_pts",
                                rr.Points2D(pts2d_gt[reproj_valid], colors=[0, 255, 0], radii=3))
                        rec.log("world/camera/image/reproj_pts",
                                rr.Points2D(uv_reproj[reproj_valid], colors=[255, 0, 0], radii=2))
                        rec.log("world/pts3d", rr.Points3D(pts3d_v, colors=[0, 180, 255], radii=0.02))

                        # Backproject depth at track positions for comparison
                        if has_depth:
                            depth = self.depths[t]
                            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                            xs = np.clip(np.round(pts2d_gt[:, 0]).astype(int), 0, W - 1)
                            ys = np.clip(np.round(pts2d_gt[:, 1]).astype(int), 0, H - 1)
                            z = depth[ys, xs].astype(np.float32)
                            dv = np.isfinite(z) & (z > 1e-3)
                            if dv.any():
                                x_c = (xs[dv] - cx) / fx * z[dv]
                                y_c = (ys[dv] - cy) / fy * z[dv]
                                pts_cam = np.stack([x_c, y_c, z[dv]], axis=1)
                                c2w_d = self.camera_poses[t].astype(np.float64)
                                pts_h = np.concatenate(
                                    [pts_cam, np.ones((len(pts_cam), 1), dtype=np.float32)], axis=1)
                                pts3d_from_depth = (c2w_d @ pts_h.T).T[:, :3].astype(np.float32)
                                rec.log("world/pts3d_from_depth",
                                        rr.Points3D(pts3d_from_depth, colors=[255, 140, 0], radii=0.02))

        print(f"  -> rerun saved: {rrd_path}")