"""
Test whether Rerun DepthImage expects z-depth or euclidean distance.
Creates a synthetic flat ground plane and logs both versions.
View with: rerun --web-viewer --port 8082 /tmp/depth_format_test.rrd
"""
import numpy as np
import rerun as rr

H, W = 200, 300
fx = fy = 250.0
cx, cy = 150.0, 100.0
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

ys, xs = np.mgrid[0:H, 0:W]
rx = (xs - cx) / fx
ry = (ys - cy) / fy
scale = np.sqrt(rx**2 + ry**2 + 1).astype(np.float32)

# Flat ground at z=5m in camera space (camera looks along +Z)
z_depth = np.full((H, W), 5.0, dtype=np.float32)
euc_depth = z_depth * scale  # euclidean distance to same flat plane

rr.init("depth_format_test", spawn=False)

# Frame 0: z-depth (constant 5.0)
rr.set_time("frame", sequence=0)
rr.log("world/cam", rr.Pinhole(image_from_camera=K, width=W, height=H))
rr.log("world/cam/depth", rr.DepthImage(z_depth, meter=1.0, colormap="Turbo", point_fill_ratio=1.0))
rr.log("info", rr.TextDocument("Frame 0: z-depth (constant 5.0m)\nShould render as FLAT plane if Rerun uses z-depth"))

# Frame 1: euclidean depth (varies, larger at edges)
rr.set_time("frame", sequence=1)
rr.log("world/cam", rr.Pinhole(image_from_camera=K, width=W, height=H))
rr.log("world/cam/depth", rr.DepthImage(euc_depth, meter=1.0, colormap="Turbo", point_fill_ratio=1.0))
rr.log("info", rr.TextDocument("Frame 1: euclidean depth (5.0~7.9m)\nShould render as FLAT plane if Rerun uses euclidean"))

rr.save("/tmp/depth_format_test.rrd")
print("Saved /tmp/depth_format_test.rrd")
print(f"Frame 0 z_depth: constant {z_depth[0,0]:.2f}m")
print(f"Frame 1 euc_depth: center={euc_depth[H//2,W//2]:.2f}m, corner={euc_depth[0,0]:.2f}m")
print()
print("View with:")
print("  conda run -n d4rt rerun --web-viewer --port 8082 /tmp/depth_format_test.rrd")
