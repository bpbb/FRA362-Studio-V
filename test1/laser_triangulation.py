import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
import csv

# =========================
# ====== PARAMETERS =======
# =========================
@dataclass
class Params:
    # Camera intrinsics (example numbers; REPLACE with yours)
    K: np.ndarray = field(default_factory=lambda: np.array([[1450.0,   0.0, 960.0],
                                                            [  0.0, 1450.0, 540.0],
                                                            [  0.0,    0.0,   1.0]], dtype=np.float64))
    dist: np.ndarray = field(default_factory=lambda: np.array([0,0,0,0,0], dtype=np.float64))

    # Laser plane in *camera* coordinates: n^T X + d = 0
    plane_n: np.ndarray = field(default_factory=lambda: np.array([0.0, -0.173648, 0.984807], dtype=np.float64))
    plane_d: float = -800.0 *3.5040 # mm; negative if plane is in front of camera along +Z

    # Optional: camera->world transform
    R_cam2world: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t_cam2world: np.ndarray = field(default_factory=lambda: np.zeros((3,1), dtype=np.float64))

    # Image path
    img_path: str = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\test\7M309979.JPG"

    # Stripe extraction
    bandpass_kernel: int = 17
    min_val_fraction: float = 0.25
    subpixel_halfwidth: int = 3

    # Depth reference
    use_reference_plane: bool = False
    ref_plane_n: np.ndarray = field(default_factory=lambda: np.array([0,0,1.0], dtype=np.float64))
    ref_plane_d: float = -800.0
    use_Z_as_height: bool = True

    # Output
    out_csv: str = "triangulated_profile.csv"


P = Params()

# =========================
# ====== UTILITIES ========
# =========================
def undistort(image, K, dist):
    if np.allclose(dist, 0):
        return image, K
    h, w = image.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
    und = cv2.undistort(image, K, dist, None, newK)
    return und, newK

def extract_laser_stripe(img_bgr, min_val_fraction=0.25, blur_k=9, sub_half=3):
    """
    Returns list of (u, v_subpix) per image column u where the stripe is found.
    - v_subpix is the sub-pixel vertical position using centroid in a (2*sub_half+1)-tall window.
    Assumes RED laser and camera sees it as strong in R channel.
    """
    red = img_bgr[:,:,2].astype(np.float32)
    if blur_k > 1:
        red = cv2.GaussianBlur(red, (blur_k, blur_k), 0)

    # threshold on fraction of max intensity
    thr = (red.max() * min_val_fraction)
    mask = (red >= thr).astype(np.uint8)

    H, W = red.shape
    rows = np.arange(H, dtype=np.float32)

    u_list, v_list = [], []
    for u in range(W):
        col = red[:,u]
        mcol = mask[:,u]
        idxs = np.where(mcol>0)[0]
        if idxs.size == 0:
            continue
        # pick brightest pixel in that column
        peak_v = int(idxs[np.argmax(col[idxs])])
        # sub-pixel centroid around peak
        v0 = max(0, peak_v - sub_half)
        v1 = min(H-1, peak_v + sub_half)
        w = col[v0:v1+1].clip(min=0.0)
        if w.sum() <= 0:
            continue
        vv = rows[v0:v1+1]
        v_c = (vv * w).sum() / w.sum()
        u_list.append(float(u))
        v_list.append(float(v_c))
    return np.array(u_list, dtype=np.float64), np.array(v_list, dtype=np.float64)

def intersect_ray_with_plane(ray_dir, n, d):
    """
    Ray originates at camera center (0,0,0), direction 'ray_dir' (3,), plane n^T X + d = 0
    Returns point X = s*ray_dir with s = -d / (n^T ray_dir). If parallel -> returns None.
    """
    denom = float(np.dot(n, ray_dir))
    if abs(denom) < 1e-9:
        return None
    s = -d / denom
    if s <= 0:
        # behind camera or at camera center
        return None
    return ray_dir * s

def backproject_pixels_to_rays(uv1, Kinv):
    """
    uv1: Nx3 homogeneous pixel coords (u,v,1)
    returns Nx3 normalized direction vectors in camera frame.
    """
    dirs = (Kinv @ uv1.T).T  # unnormalize
    # normalize direction
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return dirs / norms

def camera_to_world(Xc, R, t):
    return (R @ Xc.T + t).T

def point_plane_signed_distance(X, n, d):
    # positive if in direction of n from plane
    return (X @ n) + d

# =========================
# ======== MAIN ===========
# =========================
def main():
    img = cv2.imread(P.img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {P.img_path}")

    img_u, K = undistort(img, P.K, P.dist)
    Kinv = np.linalg.inv(K)

    # 1) Stripe extraction (sub-pixel)
    us, vs = extract_laser_stripe(
        img_u,
        min_val_fraction=P.min_val_fraction,
        blur_k=P.bandpass_kernel,
        sub_half=P.subpixel_halfwidth
    )
    if us.size == 0:
        raise RuntimeError("No laser stripe detected. Adjust thresholds/params.")

    # 2) Rays for each pixel
    ones = np.ones_like(us)
    uv1 = np.stack([us, vs, ones], axis=1)
    rays = backproject_pixels_to_rays(uv1, Kinv)  # Nx3, camera frame

    # 3) Intersect with laser plane
    Xc_list = []
    for r in rays:
        X = intersect_ray_with_plane(r, P.plane_n, P.plane_d)
        if X is not None:
            Xc_list.append(X)
        else:
            Xc_list.append([np.nan, np.nan, np.nan])
    Xc = np.array(Xc_list, dtype=np.float64)  # Nx3

    # 4) Optional: world coords
    Xw = camera_to_world(Xc, P.R_cam2world, P.t_cam2world)

    # 5) Depth/height
    if P.use_reference_plane:
        # signed distance to reference plane (camera frame)
        h = point_plane_signed_distance(Xc, P.ref_plane_n, P.ref_plane_d)
    elif P.use_Z_as_height:
        h = Xc[:,2]  # Z in camera frame (mm if your plane_d was in mm)
    else:
        h = np.zeros(Xc.shape[0])

    # 6) Save CSV
    out_path = Path(P.out_csv)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u_px","v_px","Xc(mm)","Yc(mm)","Zc(mm)","height"])
        for i in range(len(us)):
            w.writerow([us[i], vs[i], Xc[i,0], Xc[i,1], Xc[i,2], h[i]])

    print(f"Saved: {out_path.resolve()}  (rows: {len(us)})")

    # 7) Quick plots
    plt.figure()
    plt.imshow(cv2.cvtColor(img_u, cv2.COLOR_BGR2RGB))
    plt.scatter(us, vs, s=2)
    plt.title("Detected laser stripe (sub-pixel)")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.figure()
    plt.plot(us, h, linewidth=1)
    plt.xlabel("image column u (px)")
    plt.ylabel("height (mm or signed dist)")
    plt.title("Depth/height profile along the laser line")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# =========================
# ===== EXTRAS / NOTES ====
# =========================
"""
Calibrating the laser plane (quick method):

Take a flat calibration target with known 3D plane in camera frame (e.g., Z=Z0).
Capture several images while moving the laser so the line sweeps different rows.
For each image:
  - extract the stripe pixels (u,v)
  - back-project their rays to intersect with the KNOWN target plane -> 3D points on that plane
Fit a plane to those 3D points that correspond to the laser line -> that’s the LASER PLANE.
Do this at multiple tilts/distance for robustness, then average or RANSAC-fit (ax+by+cz+d=0, with ||n||=1).

Units:
- Keep everything consistent (e.g., millimeters) when setting plane_d and interpreting Z.

Robustness tips:
- Use median filter on the 1D profile to suppress outliers.
- For glossy surfaces, prefer sub-pixel via quadratic fit or Gaussian fit around the peak.
- If line gets fat, use center-of-mass across a small band, not just the brightest pixel.

To get ABSOLUTE defect “depth”:
- Set ref plane to the “ideal surface” (known or measured before defect), then report signed distance.

"""
