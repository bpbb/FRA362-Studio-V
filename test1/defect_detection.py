import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter1d, minimum_filter, median_filter, label as ndimage_label
from scipy.signal import savgol_filter
import csv

# =========================
# PARAMETERS
# =========================
@dataclass
class Params:
    # Calibration
    camera_angle_from_laser_deg: float = 30.0
    cam_to_object_distance_mm: float = 200.0
    
    # Optional scale correction
    use_scale_correction: bool = True
    correction_factor_width: float = 0.25  # For width measurements (pixel_size_mm)
    correction_factor_depth: float = 2.105 # For depth measurements (plane_d)   
    # Image
    img_path: str = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\101MSDCF\7M300001.JPG"
    
    # Camera intrinsics
    fx: float = 1450.0
    fy: float = 1450.0
    cx: float = 960.0
    cy: float = 540.0
    dist: np.ndarray = field(default_factory=lambda: np.array([0,0,0,0,0], dtype=np.float64))
    flip_image_vertical: bool = True
    
    # Laser Extraction
    laser_color: str = "blue"  # "blue", "green", "red"
    bandpass_kernel: int = 9
    subpixel_halfwidth: int = 3
    color_ratio_threshold: float = 0.49  # Adjust based on laser color
    min_val_fraction: float = 0.24      # Minimum brightness fraction

    # NEW: Initial lowpass filtering (apply first)
    enable_initial_lowpass: bool = True
    lowpass_method: str = "savgol"  # "savgol", "gaussian", "moving_avg"
    lowpass_window: int = 11        # Window size (must be odd for savgol)
    
    # Outlier removal (apply after initial lowpass)
    enable_outlier_removal: bool = True
    outlier_method: str = "improved"  # "improved", "statistical", "median", "both"
    
    # Stripe smoothing (apply last, before triangulation)
    stripe_smoothing_sigma: float = 2.0  # Reduced since we have initial lowpass
    
    # Segmentation
    auto_calculate_gap: bool = True      # Auto-calculate gap from depth data
    tomato_floor_gap_mm: float = 35.0    # Used if auto_calculate_gap=False
    min_tomato_points: int = 80
    
    # Edge-Poly Reference
    edge_region_percent: float = 25.0    # Use 25% from each edge (fallback)
    edge_poly_degree: int = 2            # Higher degree for curved surface (was 3)
    use_circular_reference: bool = True  # Use circular arc fit instead of polynomial
    defect_adjacent_points: int = 50    # Points adjacent to defect for refined reference (100)
    
    # Defect Detection
    defect_threshold_mm: float = 2.0
    min_defect_width_px: int = 10
    edge_exclude_percent: float = 10.0

    # Smoothing control
    depth_smoothing_sigma: float = 5.0   # Gaussian smoothing strength
    use_median_prefilter: bool = True    # Remove outliers first
    median_window_size: int = 5          # Window for median filter
    
    # Clustering
    cluster_defects: bool = True
    cluster_max_gap_pixels: int = 10
    
    # Output
    out_csv: str = "defects_combined.csv"
    out_plot: str = "defect_detection_combined.png"
    
    # Computed
    K: np.ndarray = field(init=False)
    plane_n: np.ndarray = field(init=False)
    plane_d: float = field(init=False)
    pixel_size_mm: float = field(init=False)  # Calculated from calibration
    
    def __post_init__(self):
        self.K = np.array([[self.fx, 0, self.cx], 
                          [0, self.fy, self.cy], 
                          [0, 0, 1]], dtype=np.float64)
        theta = np.deg2rad(self.camera_angle_from_laser_deg)
        self.plane_n = np.array([0, np.cos(theta), np.sin(theta)])
        self.plane_n /= np.linalg.norm(self.plane_n)
        
        # Base plane_d calculation
        base_plane_d = -self.cam_to_object_distance_mm * np.sin(theta)
        
        # Apply depth correction factor (affects triangulation/depth)
        if self.use_scale_correction:
            self.plane_d = base_plane_d * self.correction_factor_depth
        else:
            self.plane_d = base_plane_d
        
        # Calculate pixel size for width measurements
        avg_z = abs(self.plane_d)
        base_pixel_size = avg_z / self.fx
        
        # Apply width correction factor (affects width measurements)
        if self.use_scale_correction:
            self.pixel_size_mm = base_pixel_size * self.correction_factor_width
        else:
            self.pixel_size_mm = base_pixel_size

P = Params()


# =========================
# UTILITIES
# =========================

def pixels_to_mm(pixels, center_pixel, pixel_size_mm):
    """Convert pixel coordinates to mm relative to center."""
    return (pixels - center_pixel) * pixel_size_mm


def undistort(image, K, dist):
    """Undistort image using camera calibration."""
    if np.allclose(dist, 0):
        return image, K
    h, w = image.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
    und = cv2.undistort(image, K, dist, None, newK)
    return und, newK


# =========================
# LASER EXTRACTION
# =========================

def extract_laser_color_ratio(img_bgr, params):
    """
    Extract laser using color ratio - supports blue, green, red, and cyan lasers.
    
    The ratio is: target_color / (other_colors_sum)
    This enhances the target color relative to the background.
    """
    blue = img_bgr[:,:,0].astype(np.float32)
    green = img_bgr[:,:,1].astype(np.float32)
    red = img_bgr[:,:,2].astype(np.float32)
    
    # Apply Gaussian blur to reduce noise
    if params.bandpass_kernel > 1:
        blue = cv2.GaussianBlur(blue, (params.bandpass_kernel, params.bandpass_kernel), 0)
        green = cv2.GaussianBlur(green, (params.bandpass_kernel, params.bandpass_kernel), 0)
        red = cv2.GaussianBlur(red, (params.bandpass_kernel, params.bandpass_kernel), 0)
    
    # Select target channel and compute ratio based on laser color
    if params.laser_color == "blue":
        target = blue
        color_ratio = blue / (red + green + 1e-6)
        
    elif params.laser_color == "green":
        target = green
        color_ratio = (blue + green) / (0.5 * red + 1e-6)
        
    elif params.laser_color == "red":
        target = red
        color_ratio = red / (blue + green + 1e-6)
        
    else:
        raise ValueError(f"Unknown laser color: {params.laser_color}. Use 'blue', 'green', 'red'.")
    
    # Two-stage thresholding: brightness AND color ratio
    brightness_threshold = target.max() * params.min_val_fraction
    ratio_threshold = params.color_ratio_threshold
    
    mask = (target >= brightness_threshold) & (color_ratio >= ratio_threshold)
    mask = mask.astype(np.uint8)
    
    H, W = target.shape
    rows = np.arange(H, dtype=np.float32)
    
    u_list, v_list = [], []
    for u in range(W):
        col = target[:,u]
        mcol = mask[:,u]
        idxs = np.where(mcol > 0)[0]
        if idxs.size == 0:
            continue
        
        # Find peak within mask
        peak_v = int(idxs[np.argmax(col[idxs])])
        
        # Subpixel refinement
        v0 = max(0, peak_v - params.subpixel_halfwidth)
        v1 = min(H-1, peak_v + params.subpixel_halfwidth)
        w = col[v0:v1+1].clip(min=0.0)
        if w.sum() <= 0:
            continue
        
        vv = rows[v0:v1+1]
        v_c = (vv * w).sum() / w.sum()
        
        # Handle vertical flip if needed
        if params.flip_image_vertical:
            v_c = H - 1 - v_c
        
        u_list.append(float(u))
        v_list.append(float(v_c))
    
    return np.array(u_list), np.array(v_list)


# =========================
# STRIPE PREPROCESSING
# =========================

def initial_lowpass_filter(us, vs, method="savgol", window=11):
    """
    Apply initial lowpass filter to raw stripe data.
    This removes high-frequency noise BEFORE outlier detection.
    
    This is the FIRST cleaning stage.
    
    Methods:
        "savgol": Savitzky-Golay filter (preserves shape while smoothing)
        "gaussian": Gaussian filter (simple and effective)
        "moving_avg": Moving average (fastest, most smoothing)
    """
    if len(us) < window:
        print(f"\nInitial lowpass: Skipped (too few points: {len(us)} < {window})")
        return us, vs
    
    print(f"\n{'='*60}")
    print(f"STAGE 1: Initial Lowpass Filtering")
    print(f"{'='*60}")
    print(f"   Method: {method}")
    print(f"   Window: {window} points")
    print(f"   Input points: {len(us)}")
    
    # Store original for comparison
    vs_original = vs.copy()
    
    if method == "savgol":
        # Savitzky-Golay filter: fits polynomial locally
        # Best for preserving features while removing noise
        polyorder = min(3, window - 2)  # Polynomial order < window
        if window % 2 == 0:
            window += 1  # Must be odd
        vs_filtered = savgol_filter(vs, window, polyorder, mode='nearest')
        
    elif method == "gaussian":
        # Convert window to sigma (window ≈ 6*sigma)
        sigma = window / 6.0
        vs_filtered = gaussian_filter1d(vs, sigma=sigma, mode='nearest')
        
    elif method == "moving_avg":
        # Simple moving average
        from scipy.ndimage import uniform_filter1d
        vs_filtered = uniform_filter1d(vs, size=window, mode='nearest')
    else:
        raise ValueError(f"Unknown filter method: {method}")
    
    # Calculate noise reduction metrics
    noise_removed = np.std(vs_original - vs_filtered)
    max_change = np.max(np.abs(vs_original - vs_filtered))
    print(f"   Noise reduced: {noise_removed:.2f} px RMS")
    print(f"   Max change: {max_change:.2f} px")
    
    return us, vs_filtered


def remove_outliers_from_stripe(us, vs, method="improved", params=None):
    """
    Remove outliers from extracted laser stripe.
    This is the SECOND cleaning stage (after initial lowpass).
    
    Methods:
        "improved": Multi-stage robust outlier removal (RECOMMENDED)
        "statistical": Remove points beyond 3 sigma gradient
        "median": Remove points far from local median
        "both": Apply both statistical and median
    
    Returns:
        us_clean, vs_clean: Arrays with outliers removed
    """
    
    if len(us) < 10:
        print(f"\nOutlier removal: Skipped (too few points: {len(us)})")
        return us, vs
    
    print(f"\n{'='*60}")
    print(f"STAGE 2: Outlier Removal ({method})")
    print(f"{'='*60}")
    print(f"   Input points: {len(us)}")
    
    initial_count = len(us)
    
    if method == "improved":
        # ====== STAGE 2.1: Remove large jumps (discontinuities) ======
        dv = np.diff(vs)
        dv_pad = np.concatenate([[0], dv])
        
        # Use robust statistics (MAD)
        mad = np.median(np.abs(dv - np.median(dv)))
        threshold_jump = 2.0 * (mad * 1.4826)  # 2-sigma for large jumps
        
        mask_continuous = np.abs(dv_pad) < threshold_jump
        outliers_stage1 = (~mask_continuous).sum()
        
        if outliers_stage1 > 0:
            us = us[mask_continuous]
            vs = vs[mask_continuous]
            print(f"   2.1 Large jumps: removed {outliers_stage1} points")
        else:
            print(f"   2.1 Large jumps: no outliers found")
        
        if len(us) < 10:
            print(f"   WARNING: Too few points remaining ({len(us)})")
            return us, vs
        
        # ====== STAGE 2.2: Local median absolute deviation ======
        window = 31  # Larger window for robust local median
        half_window = window // 2
        
        outlier_mask = np.ones(len(vs), dtype=bool)
        
        for i in range(len(vs)):
            start = max(0, i - half_window)
            end = min(len(vs), i + half_window + 1)
            
            local_window = vs[start:end]
            local_median = np.median(local_window)
            local_mad = np.median(np.abs(local_window - local_median))
            
            # Adaptive threshold based on local variation
            if local_mad < 0.1:  # Very stable region
                threshold = 2.0
            else:
                threshold = 3.5 * (local_mad * 1.4826)
            
            if np.abs(vs[i] - local_median) > threshold:
                outlier_mask[i] = False
        
        outliers_stage2 = (~outlier_mask).sum()
        if outliers_stage2 > 0:
            us = us[outlier_mask]
            vs = vs[outlier_mask]
            print(f"   2.2 Local MAD: removed {outliers_stage2} points")
        else:
            print(f"   2.2 Local MAD: no outliers found")
        
        if len(us) < 10:
            print(f"   WARNING: Too few points remaining ({len(us)})")
            return us, vs
        
        # ====== STAGE 2.3: Isolated point removal ======
        # Remove points that are alone (large gaps on both sides)
        gaps = np.diff(us)
        gap_threshold = np.median(gaps) + 3 * np.std(gaps)
        
        isolated_mask = np.ones(len(us), dtype=bool)
        for i in range(1, len(us) - 1):
            if gaps[i-1] > gap_threshold and gaps[i] > gap_threshold:
                isolated_mask[i] = False
        
        outliers_stage3 = (~isolated_mask).sum()
        if outliers_stage3 > 0:
            us = us[isolated_mask]
            vs = vs[isolated_mask]
            print(f"   2.3 Isolated points: removed {outliers_stage3} points")
        else:
            print(f"   2.3 Isolated points: no outliers found")
    
    elif method == "statistical":
        # Calculate gradient (change in v)
        dv = np.diff(vs)
        dv = np.concatenate([[0], dv])
        
        # Remove points with excessive gradient
        median_dv = np.median(np.abs(dv))
        mad = np.median(np.abs(dv - np.median(dv)))
        threshold = median_dv + 3 * (mad * 1.4826)
        
        mask_grad = np.abs(dv) < threshold
        
        outliers_removed = (~mask_grad).sum()
        print(f"   Statistical: removed {outliers_removed} points")
        
        us = us[mask_grad]
        vs = vs[mask_grad]
    
    elif method == "median":
        # Use rolling median to detect outliers
        window = 21
        half_window = window // 2
        
        outlier_mask = np.ones(len(vs), dtype=bool)
        
        for i in range(len(vs)):
            start = max(0, i - half_window)
            end = min(len(vs), i + half_window + 1)
            
            local_window = vs[start:end]
            local_median = np.median(local_window)
            local_mad = np.median(np.abs(local_window - local_median))
            
            threshold = 3 * (local_mad * 1.4826)
            
            if np.abs(vs[i] - local_median) > threshold:
                outlier_mask[i] = False
        
        outliers_removed = (~outlier_mask).sum()
        print(f"   Median: removed {outliers_removed} points")
        
        us = us[outlier_mask]
        vs = vs[outlier_mask]
    
    elif method == "both":
        # Apply statistical first
        dv = np.diff(vs)
        dv = np.concatenate([[0], dv])
        
        median_dv = np.median(np.abs(dv))
        mad = np.median(np.abs(dv - np.median(dv)))
        threshold = median_dv + 3 * (mad * 1.4826)
        
        mask_grad = np.abs(dv) < threshold
        outliers_removed = (~mask_grad).sum()
        print(f"   Statistical: removed {outliers_removed} points")
        
        us = us[mask_grad]
        vs = vs[mask_grad]
        
        # Then apply median
        window = 21
        half_window = window // 2
        outlier_mask = np.ones(len(vs), dtype=bool)
        
        for i in range(len(vs)):
            start = max(0, i - half_window)
            end = min(len(vs), i + half_window + 1)
            
            local_window = vs[start:end]
            local_median = np.median(local_window)
            local_mad = np.median(np.abs(local_window - local_median))
            
            threshold = 3 * (local_mad * 1.4826)
            
            if np.abs(vs[i] - local_median) > threshold:
                outlier_mask[i] = False
        
        outliers_removed = (~outlier_mask).sum()
        print(f"   Median: removed {outliers_removed} points")
        
        us = us[outlier_mask]
        vs = vs[outlier_mask]
    
    total_removed = initial_count - len(us)
    removal_percent = 100 * total_removed / initial_count
    print(f"   Output points: {len(us)} ({removal_percent:.1f}% removed)")
    
    return us, vs


def smooth_stripe_positions(us, vs, sigma=2.0):
    """
    Apply final smoothing to stripe positions with Gaussian filter.
    This is the THIRD cleaning stage (after lowpass and outlier removal).
    
    Should use lower sigma than before since we already have initial lowpass.
    """
    
    if sigma <= 0 or len(us) < 5:
        print(f"\nFinal smoothing: Skipped (sigma={sigma})")
        return us, vs
    
    print(f"\n{'='*60}")
    print(f"STAGE 3: Final Stripe Smoothing")
    print(f"{'='*60}")
    print(f"   Gaussian filter: sigma={sigma:.1f}")
    
    # Store original for comparison
    vs_original = vs.copy()
    
    # Smooth v-coordinates
    vs_smooth = gaussian_filter1d(vs, sigma=sigma, mode='nearest')
    
    # Calculate smoothing effect
    smoothing_change = np.std(vs_original - vs_smooth)
    max_change = np.max(np.abs(vs_original - vs_smooth))
    print(f"   RMS change: {smoothing_change:.2f} px")
    print(f"   Max change: {max_change:.2f} px")
    
    return us, vs_smooth


# =========================
# TRIANGULATION
# =========================

def triangulate(us, vs, params):
    """Triangulate 3D points from 2D laser stripe."""
    ones = np.ones_like(us)
    uv1 = np.stack([us, vs, ones], axis=1)
    
    Kinv = np.linalg.inv(params.K)
    rays = (Kinv @ uv1.T).T
    rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12)
    
    depths = []
    for ray in rays:
        denom = np.dot(params.plane_n, ray)
        if abs(denom) > 1e-9:
            t = -params.plane_d / denom
            if t > 0:
                depths.append(ray[2] * t)
                continue
        depths.append(np.nan)
    
    return np.array(depths)


# =========================
# DETREND FLOOR TILT
# =========================

def detrend_floor_tilt(us, depths, method="linear"):
    """
    Remove floor tilt by fitting and subtracting a trend line.
    
    Methods:
        "linear": Fit straight line to edges (fast, good for small tilt)
        "polynomial": Fit polynomial to edges (better for curved floors)
        "robust": RANSAC-based fit (best for noisy data)
    """
    n = len(depths)
    edge_size = max(50, int(n * 0.1))  # Use 10% from each edge
    
    # Extract floor points (edges assumed to be floor)
    floor_indices = np.concatenate([
        np.arange(edge_size),
        np.arange(n - edge_size, n)
    ])
    floor_us = us[floor_indices]
    floor_depths = depths[floor_indices]
    
    print(f"\nDetrending floor tilt:")
    print(f"   Using {len(floor_indices)} floor points from edges")
    
    if method == "linear":
        # Fit linear trend: depth = a*u + b
        coeffs = np.polyfit(floor_us, floor_depths, deg=1)
        trend = np.polyval(coeffs, us)
        
        tilt_angle = np.rad2deg(np.arctan(coeffs[0]))
        print(f"   Linear fit: slope={coeffs[0]:.6f} mm/px ({tilt_angle:.3f} deg)")
        print(f"   Tilt across image: {coeffs[0] * n:.1f} mm")
        
    elif method == "polynomial":
        # Fit quadratic for curved floors
        coeffs = np.polyfit(floor_us, floor_depths, deg=2)
        trend = np.polyval(coeffs, us)
        print(f"   Polynomial fit (degree 2)")
        
    elif method == "robust":
        # RANSAC for robust fitting with outliers
        from sklearn.linear_model import RANSACRegressor
        
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(floor_us.reshape(-1, 1), floor_depths)
        trend = ransac.predict(us.reshape(-1, 1))
        print(f"   RANSAC robust fit")
    
    # Subtract the trend (normalize to median floor level)
    floor_median = np.median(floor_depths)
    depths_detrended = depths - trend + floor_median
    
    # Show correction statistics
    correction_range = trend.max() - trend.min()
    print(f"   Removed tilt: {correction_range:.1f} mm range")
    print(f"   New floor level: {floor_median:.1f} mm (horizontal)")
    
    return depths_detrended, trend, floor_median


# =========================
# REFERENCE SURFACE FITTING
# =========================

def fit_circular_arc(us, depths):
    """
    Fit a circular arc to points (u, depth).
    Returns reference depths for all u values.
    
    Circle equation: (u - u_c)^2 + (d - d_c)^2 = R^2
    We fit to find center (u_c, d_c) and radius R.
    """
    from scipy.optimize import least_squares
    
    # Initial guess: center at middle, radius from span
    u_mid = (us.max() + us.min()) / 2
    d_mid = (depths.max() + depths.min()) / 2
    r_init = max(us.max() - us.min(), depths.max() - depths.min()) / 2
    
    def residuals(params):
        u_c, d_c, R = params
        return np.sqrt((us - u_c)**2 + (depths - d_c)**2) - R
    
    # Fit circle
    result = least_squares(residuals, [u_mid, d_mid, r_init])
    u_c, d_c, R = result.x
    
    # Generate arc for all u values in range
    u_all = np.arange(us.min(), us.max() + 1)
    
    # Calculate corresponding depths on circle
    # (u - u_c)^2 + (d - d_c)^2 = R^2
    # d = d_c ± sqrt(R^2 - (u - u_c)^2)
    
    under_sqrt = R**2 - (u_all - u_c)**2
    valid = under_sqrt >= 0
    
    # Choose the arc that matches the data (upper or lower arc)
    d_all = np.full(len(u_all), np.nan)
    
    if valid.any():
        # Try both arcs and pick the one closer to data
        d_upper = d_c + np.sqrt(under_sqrt[valid])
        d_lower = d_c - np.sqrt(under_sqrt[valid])
        
        # Check which arc is closer to original data
        if depths.mean() > d_c:
            d_all[valid] = d_upper
        else:
            d_all[valid] = d_lower
    
    return u_all, d_all, (u_c, d_c, R)


# =========================
# SEGMENTATION
# =========================

def auto_calculate_tomato_gap(depths, verbose=True):
    """
    Automatically calculate optimal tomato_floor_gap_mm based on depth data.
    
    Uses the same algorithm as the tuning tool:
    1. Calculate depth range
    2. Initial estimate: 20% of range
    3. Test segmentation
    4. Adjust based on tomato coverage
    
    Args:
        depths: Array of depth values (should be detrended)
        verbose: Print calculation details
    
    Returns:
        float: Recommended gap value in mm
    """
    n = len(depths)
    
    # Estimate floor level from edges
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths[:edge_size], depths[-edge_size:]]))
    
    # Calculate depth range
    depth_min = depths.min()
    depth_max = depths.max()
    depth_range = depth_max - depth_min
    
    # Initial estimate: 20% of depth range
    initial_gap = depth_range * 0.20
    
    # Test segmentation with initial gap
    tomato_threshold = floor_level - initial_gap
    tomato_mask_test = depths < tomato_threshold
    tomato_count = tomato_mask_test.sum()
    tomato_percent = 100 * tomato_count / n
    
    # Adjust based on coverage
    if tomato_percent < 10:
        # Too few tomato points
        recommended_gap = depth_range * 0.18
        status = "Too few tomato points - decreased gap"
    elif tomato_percent > 70:
        # Too many tomato points
        recommended_gap = depth_range * 0.40
        status = "Too many tomato points - increased gap"
    elif tomato_percent < 25:
        # Low coverage
        recommended_gap = depth_range * 0.22
        status = "Low tomato coverage - slightly decreased gap"
    elif tomato_percent > 55:
        # High coverage
        recommended_gap = depth_range * 0.32
        status = "High tomato coverage - slightly increased gap"
    else:
        # Good coverage - use higher baseline
        recommended_gap = depth_range * 0.27
        status = "Good segmentation"
    
    if verbose:
        print(f"\nAuto-calculating tomato_floor_gap_mm:")
        print(f"   Depth range: {depth_range:.1f} mm")
        print(f"   Floor level: {floor_level:.1f} mm")
        print(f"   Initial gap (20%): {initial_gap:.1f} mm")
        print(f"   Test coverage: {tomato_percent:.1f}% tomato points")
        print(f"   Status: {status}")
        print(f"   Recommended gap: {recommended_gap:.1f} mm")
    
    return recommended_gap


def segment_tomato(depths, params):
    """
    Segment tomato from floor using simple threshold.
    Uses the SAME logic as parameter tuning tool.
    
    If params.auto_calculate_gap is True, automatically calculates
    optimal gap from depth data.
    
    Returns tomato mask and estimated floor level.
    """
    n = len(depths)
    
    # Estimate floor level from edges (same as parameter tuning)
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths[:edge_size], depths[-edge_size:]]))
    
    # Auto-calculate gap if enabled
    if params.auto_calculate_gap:
        gap_mm = auto_calculate_tomato_gap(depths, verbose=True)
    else:
        gap_mm = params.tomato_floor_gap_mm
        print(f"\nSegmentation:")
        print(f"   Using fixed gap: {gap_mm:.1f} mm")
    
    print(f"   Floor level: {floor_level:.1f} mm")
    print(f"   Depth range: {depths.min():.1f} - {depths.max():.1f} mm")
    print(f"   Using gap threshold: {gap_mm:.1f} mm")
    
    # Simple threshold: tomato is closer to camera (smaller depth)
    tomato_mask = depths < (floor_level - gap_mm)
    
    # Keep only largest connected component to remove small noise
    labeled, num = ndimage_label(tomato_mask)
    
    if num > 0:
        # Find largest component
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        largest = np.argmax(sizes) + 1
        tomato_mask = (labeled == largest)
        
        if tomato_mask.sum() < params.min_tomato_points:
            print(f"   WARNING: Too few points ({tomato_mask.sum()}), rejecting")
            tomato_mask = np.zeros(n, dtype=bool)
        else:
            tomato_percent = 100 * tomato_mask.sum() / n
            print(f"   Tomato: {tomato_mask.sum()} points ({tomato_percent:.1f}%)")
    else:
        tomato_mask = np.zeros(n, dtype=bool)
        print(f"   WARNING: No tomato region found")
    
    return tomato_mask, floor_level


# =========================
# EDGE-POLY REFERENCE
# =========================

def create_defect_adjacent_reference(us, depths, params, rough_defect_mask=None):
    """
    Create reference surface using healthy edges ADJACENT to defects.
    
    If rough_defect_mask is provided, uses local interpolation between
    points near defect boundaries. Otherwise, falls back to edge-based reference.
    
    IMPROVED: Better edge detection for small defects with validation
    
    Args:
        us: Pixel coordinates
        depths: Depth values
        params: Parameters
        rough_defect_mask: Optional boolean mask of rough defect locations
    
    Returns:
        reference: Fitted reference surface
        edge_mask: Boolean mask of points used for fitting
        degree_name: Name of polynomial degree
    """
    n = len(us)
    
    if rough_defect_mask is not None and rough_defect_mask.any():
        # Use defect-adjacent approach with IMPROVED LOCAL fitting
        
        edge_mask = np.zeros(n, dtype=bool)
        reference = np.full(n, np.nan)
        
        # Find defect regions
        labeled, num_defects = ndimage_label(rough_defect_mask)
        
        print(f"\nReference Surface (Defect-Adjacent):")
        print(f"   Method: Local interpolation using validated defect edges")
        
        total_edge_points = 0
        valid_defects = 0
        
        for i in range(1, num_defects + 1):
            defect_region = (labeled == i)
            defect_indices = np.where(defect_region)[0]
            
            if len(defect_indices) == 0:
                continue
            
            start_idx = defect_indices[0]
            end_idx = defect_indices[-1]
            defect_width = end_idx - start_idx + 1
            
            # IMPROVEMENT: Adaptive margin based on defect size
            # Larger defects need wider margins to find healthy tissue
            base_margin = params.defect_adjacent_points
            if defect_width < 20:
                # Small defect: reduce margin to avoid including distant regions
                edge_margin = max(15, base_margin // 2)
            elif defect_width < 50:
                # Medium defect: use standard margin
                edge_margin = base_margin
            else:
                # Large defect: increase margin for better context
                edge_margin = int(base_margin * 1.5)
            
            # Get candidate edge points BEFORE defect
            left_start = max(0, start_idx - edge_margin)
            left_end = start_idx
            left_indices = np.arange(left_start, left_end)
            
            # Get candidate edge points AFTER defect
            right_start = end_idx + 1
            right_end = min(n, end_idx + 1 + edge_margin)
            right_indices = np.arange(right_start, right_end)
            
            # IMPROVEMENT: Validate that edge regions are actually healthier than defect
            # Check that edge depths are lower (closer to camera = healthier)
            defect_mean_depth = depths[defect_indices].mean()
            
            valid_left = False
            valid_right = False
            
            if len(left_indices) > 5:
                left_depths = depths[left_indices]
                left_mean_depth = left_depths.mean()
                left_std = left_depths.std()
                
                # Check for edge spikes (rapid depth changes = triangulation errors)
                left_gradients = np.abs(np.diff(left_depths))
                max_gradient = left_gradients.max() if len(left_gradients) > 0 else 0
                
                # Edge must be: healthier, stable, AND no spikes
                if (left_mean_depth < defect_mean_depth - 0.5 and 
                    left_std < 3.0 and 
                    max_gradient < 5.0):
                    valid_left = True
                else:
                    # Try smaller margin, skip boundary points if they have spikes
                    edge_margin_small = max(10, edge_margin // 2)
                    left_start_new = max(0, start_idx - edge_margin_small)
                    # If at left boundary, skip first few points
                    if left_start == 0 and left_end > 10:
                        left_start_new = min(5, left_end - 10)
                    left_indices = np.arange(left_start_new, left_end)
                    if len(left_indices) > 5:
                        left_depths = depths[left_indices]
                        left_mean_depth = left_depths.mean()
                        left_std = left_depths.std()
                        left_gradients = np.abs(np.diff(left_depths))
                        max_gradient = left_gradients.max() if len(left_gradients) > 0 else 0
                        if (left_mean_depth < defect_mean_depth - 0.5 and 
                            left_std < 3.0 and 
                            max_gradient < 5.0):
                            valid_left = True
            
            if len(right_indices) > 5:
                right_depths = depths[right_indices]
                right_mean_depth = right_depths.mean()
                right_std = right_depths.std()
                
                # Check for edge spikes
                right_gradients = np.abs(np.diff(right_depths))
                max_gradient = right_gradients.max() if len(right_gradients) > 0 else 0
                
                # Edge must be: healthier, stable, AND no spikes
                if (right_mean_depth < defect_mean_depth - 0.5 and 
                    right_std < 3.0 and 
                    max_gradient < 5.0):
                    valid_right = True
                else:
                    # Try smaller margin, skip boundary points if they have spikes
                    edge_margin_small = max(10, edge_margin // 2)
                    right_end_new = min(n, end_idx + 1 + edge_margin_small)
                    # If at right boundary, skip last few points
                    if right_end == n and right_start < n - 10:
                        right_end_new = max(right_start + 5, n - 5)
                    right_indices = np.arange(right_start, right_end_new)
                    if len(right_indices) > 5:
                        right_depths = depths[right_indices]
                        right_mean_depth = right_depths.mean()
                        right_std = right_depths.std()
                        right_gradients = np.abs(np.diff(right_depths))
                        max_gradient = right_gradients.max() if len(right_gradients) > 0 else 0
                        if (right_mean_depth < defect_mean_depth - 0.5 and 
                            right_std < 3.0 and 
                            max_gradient < 5.0):
                            valid_right = True
            
            # IMPROVEMENT: Only use this defect if we have valid edges
            if not (valid_left or valid_right):
                print(f"   Defect {i}: No valid edges (width={defect_width}px, at boundary or edges unhealthy)")
                continue
            
            # Mark valid edge regions
            if valid_left and len(left_indices) > 0:
                edge_mask[left_indices] = True
                total_edge_points += len(left_indices)
            
            if valid_right and len(right_indices) > 0:
                edge_mask[right_indices] = True
                total_edge_points += len(right_indices)
            
            # Create reference for THIS defect region using local fitting
            if valid_left and valid_right:
                # Both edges available: polynomial interpolation
                edge_indices = np.concatenate([left_indices, right_indices])
                edge_us = us[edge_indices]
                edge_depths = depths[edge_indices]
                
                # Fit polynomial to JUST these local edge points
                if len(edge_us) >= params.edge_poly_degree + 1:
                    coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                    
                    # Evaluate reference ONLY in defect region + edges
                    local_region = np.arange(left_start, right_end)
                    reference[local_region] = np.polyval(coeffs, us[local_region])
                    valid_defects += 1
                else:
                    # Fall back to linear interpolation
                    left_mean = depths[left_indices].mean()
                    right_mean = depths[right_indices].mean()
                    
                    interp_region = np.arange(start_idx, end_idx + 1)
                    t = (us[interp_region] - us[start_idx]) / (us[end_idx] - us[start_idx] + 1e-9)
                    reference[interp_region] = left_mean + t * (right_mean - left_mean)
                    valid_defects += 1
                    
            elif valid_left:
                # Only left edge: extend leftward
                left_mean = depths[left_indices].mean()
                local_region = np.arange(left_start, end_idx + 1)
                reference[local_region] = left_mean
                valid_defects += 1
                
            elif valid_right:
                # Only right edge: extend rightward
                right_mean = depths[right_indices].mean()
                local_region = np.arange(start_idx, right_end)
                reference[local_region] = right_mean
                valid_defects += 1
        
        print(f"   Valid defects with healthy edges: {valid_defects}/{num_defects}")
        print(f"   Edge points: {total_edge_points}/{n} ({100*total_edge_points/n:.1f}%)")
        
        # Set degree_name for defect-adjacent method
        degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
            params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
        )
        
        # IMPROVEMENT: Fall back to standard edges if too few valid edge points
        if total_edge_points < 20 or valid_defects == 0:
            print(f"   WARNING: Insufficient validated edges, falling back to standard edges")
            # Fall back to standard edge-based reference
            edge_size = max(5, int(n * params.edge_region_percent / 100.0))
            edge_mask = np.zeros(n, dtype=bool)
            edge_mask[:edge_size] = True
            edge_mask[-edge_size:] = True
            
            edge_us = us[edge_mask]
            edge_depths = depths[edge_mask]
            
            # Filter out boundary spikes before fitting
            edge_gradients = np.abs(np.diff(edge_depths))
            if len(edge_gradients) > 0:
                spike_threshold = 5.0  # mm
                # Find first and last good points (no large gradients)
                good_start = 0
                for i in range(min(5, len(edge_gradients))):
                    if edge_gradients[i] > spike_threshold:
                        good_start = i + 2  # Skip spike
                    else:
                        break
                
                good_end = len(edge_depths)
                for i in range(max(0, len(edge_gradients) - 5), len(edge_gradients)):
                    if edge_gradients[i] > spike_threshold:
                        good_end = i  # Exclude from this point
                        break
                
                # Keep only good region
                if good_start < good_end and (good_end - good_start) > 10:
                    edge_us = edge_us[good_start:good_end]
                    edge_depths = edge_depths[good_start:good_end]
                    print(f"   Filtered boundary spikes: using points {good_start} to {good_end}")
            
            if params.use_circular_reference and len(edge_us) >= 10:
                try:
                    u_arc, d_arc, (u_c, d_c, R) = fit_circular_arc(edge_us, edge_depths)
                    reference = np.interp(us, u_arc, d_arc, left=np.nan, right=np.nan)
                    degree_name = "circular arc"
                except:
                    coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                    reference = np.polyval(coeffs, us)
            else:
                coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                reference = np.polyval(coeffs, us)
    
    else:
        # Fall back to standard edge-based reference
        edge_size = max(5, int(n * params.edge_region_percent / 100.0))
        edge_mask = np.zeros(n, dtype=bool)
        edge_mask[:edge_size] = True
        edge_mask[-edge_size:] = True
        
        print(f"\nReference Surface (Standard Edges):")
        
        edge_us = us[edge_mask]
        edge_depths = depths[edge_mask]
        
        # Filter out boundary spikes before fitting
        edge_gradients = np.abs(np.diff(edge_depths))
        if len(edge_gradients) > 0:
            spike_threshold = 5.0  # mm
            # Find first and last good points (no large gradients)
            good_start = 0
            for i in range(min(5, len(edge_gradients))):
                if edge_gradients[i] > spike_threshold:
                    good_start = i + 2  # Skip spike
                else:
                    break
            
            good_end = len(edge_depths)
            for i in range(max(0, len(edge_gradients) - 5), len(edge_gradients)):
                if edge_gradients[i] > spike_threshold:
                    good_end = i  # Exclude from this point
                    break
            
            # Keep only good region
            if good_start < good_end and (good_end - good_start) > 10:
                edge_us = edge_us[good_start:good_end]
                edge_depths = edge_depths[good_start:good_end]
                print(f"   Filtered boundary spikes: using points {good_start} to {good_end}")
        
        if params.use_circular_reference and len(edge_us) >= 10:
            # Fit circular arc to edges
            print(f"   Method: Circular arc fit to tomato edges")
            print(f"   Edge regions: {edge_size*2}/{n} points ({params.edge_region_percent*2:.0f}%)")
            
            try:
                u_arc, d_arc, (u_c, d_c, R) = fit_circular_arc(edge_us, edge_depths)
                
                # Interpolate arc to all u values
                reference = np.interp(us, u_arc, d_arc, left=np.nan, right=np.nan)
                
                print(f"   Circle: center=({u_c:.1f}, {d_c:.1f}), radius={R:.1f}mm")
                degree_name = "circular arc"
                
            except Exception as e:
                print(f"   WARNING: Circle fit failed ({e}), using polynomial")
                coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                reference = np.polyval(coeffs, us)
                degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
                    params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
                )
        else:
            # Use polynomial fit
            print(f"   Method: Polynomial (degree {params.edge_poly_degree}) fit to edges")
            print(f"   Edge regions: {edge_size*2}/{n} points ({params.edge_region_percent*2:.0f}%)")
            
            coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
            reference = np.polyval(coeffs, us)
            degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
                params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
            )
    
    return reference, edge_mask, degree_name


# =========================
# DEFECT DETECTION
# =========================

def detect_defects(us, depths, tomato_mask, params):
    """
    Detect defects using two-pass defect-adjacent reference.
    
    Pass 1: Rough detection using standard edge-based reference
    Pass 2: Refined detection using points adjacent to defects
    
    Returns list of defects and various masks/arrays for visualization.
    """
    # Extract tomato data
    us_tomato = us[tomato_mask]
    depths_tomato = depths[tomato_mask]
    
    if len(us_tomato) < 10:
        return [], np.zeros(len(us), dtype=bool), np.full(len(us), np.nan), None, None
    
    # Pre-smooth the depths
    if params.depth_smoothing_sigma > 0:
        print(f"\nPre-smoothing depths:")
        
        if params.use_median_prefilter:
            window = int(params.median_window_size)
            if window % 2 == 0:
                window += 1
            
            depths_clean = median_filter(
                depths_tomato, 
                size=window,
                mode='nearest'
            )
            print(f"   Median filter: window={window}")
        else:
            depths_clean = depths_tomato.copy()
        
        depths_processed = gaussian_filter1d(
            depths_clean,
            sigma=params.depth_smoothing_sigma,
            mode='nearest'
        )
        print(f"   Gaussian smoothing: sigma={params.depth_smoothing_sigma:.1f}")
    else:
        depths_processed = depths_tomato.copy()
        print(f"\nNo pre-smoothing applied")
    
    # =================================================================
    # PASS 1: Rough defect detection using standard edge-based reference
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PASS 1: Initial defect detection")
    print(f"{'='*60}")
    
    ref_rough, edge_mask_rough, _ = create_defect_adjacent_reference(
        us_tomato, depths_processed, params, rough_defect_mask=None
    )
    
    dev_rough = depths_processed - ref_rough
    
    # Apply light smoothing to deviation
    if params.depth_smoothing_sigma > 0:
        deviation_sigma = params.depth_smoothing_sigma * 0.5
        if deviation_sigma > 0:
            dev_rough = gaussian_filter1d(dev_rough, sigma=deviation_sigma, mode='nearest')
    
    # Initial defect detection
    rough_defect_mask = dev_rough > params.defect_threshold_mm
    
    # Clean up: remove edge regions and small isolated regions
    if params.edge_exclude_percent > 0:
        edge_margin = int(len(depths_tomato) * params.edge_exclude_percent / 100.0)
        if edge_margin > 0:
            rough_defect_mask[:edge_margin] = False
            rough_defect_mask[-edge_margin:] = False
    
    labeled_rough, num_rough = ndimage_label(rough_defect_mask)
    
    # Keep only significant regions
    for i in range(1, num_rough + 1):
        if (labeled_rough == i).sum() < params.min_defect_width_px:
            rough_defect_mask[labeled_rough == i] = False
    
    num_rough_defects = (ndimage_label(rough_defect_mask)[1])
    print(f"   Initial defects found: {num_rough_defects}")
    
    if num_rough_defects == 0:
        print(f"   No defects detected, using standard reference")
        ref_tomato = ref_rough
        edge_mask_tomato = edge_mask_rough
        dev_tomato = dev_rough
        defect_mask_tomato = rough_defect_mask
        degree_name = {1: "linear", 2: "quadratic", 3: "cubic"}.get(
            params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
        )
    else:
        # =================================================================
        # PASS 2: Refined detection using defect-adjacent reference
        # =================================================================
        print(f"\n{'='*60}")
        print(f"PASS 2: Refined defect detection")
        print(f"{'='*60}")
        
        ref_tomato, edge_mask_tomato, degree_name = create_defect_adjacent_reference(
            us_tomato, depths_processed, params, rough_defect_mask=rough_defect_mask
        )
        
        # Recalculate deviations with better reference
        dev_tomato = depths_processed - ref_tomato
        
        # Apply smoothing to refined deviation
        if params.depth_smoothing_sigma > 0:
            deviation_sigma = params.depth_smoothing_sigma * 0.5
            if deviation_sigma > 0:
                dev_tomato = gaussian_filter1d(dev_tomato, sigma=deviation_sigma, mode='nearest')
                print(f"   Deviation smoothing: sigma={deviation_sigma:.1f}")
        
        # Final defect detection with refined reference
        defect_mask_tomato = dev_tomato > params.defect_threshold_mm
        
        # Exclude edges
        if params.edge_exclude_percent > 0:
            edge_margin = int(len(depths_tomato) * params.edge_exclude_percent / 100.0)
            if edge_margin > 0:
                defect_mask_tomato[:edge_margin] = False
                defect_mask_tomato[-edge_margin:] = False
    
    # Map back to full arrays
    reference = np.full(len(us), np.nan)
    reference[tomato_mask] = ref_tomato
    
    defect_mask_full = np.zeros(len(us), dtype=bool)
    defect_mask_full[tomato_mask] = defect_mask_tomato
    
    edge_mask_full = np.zeros(len(us), dtype=bool)
    edge_mask_full[tomato_mask] = edge_mask_tomato
    
    # Cluster defects into distinct regions
    defects = []
    
    if params.cluster_defects:
        labeled, num = ndimage_label(defect_mask_tomato)
        
        print(f"\n{'='*60}")
        print(f"FINAL DEFECT DETECTION")
        print(f"{'='*60}")
        print(f"   Threshold: {params.defect_threshold_mm:.2f} mm")
        print(f"   Min width: {params.min_defect_width_px} px")
        print(f"   Initial regions: {num}")
        
        for i in range(1, num + 1):
            region = (labeled == i)
            
            if region.sum() < params.min_defect_width_px:
                continue
            
            indices = np.where(region)[0]
            defect_depths = dev_tomato[region]
            
            defects.append({
                'id': len(defects) + 1,
                'start_px': int(us_tomato[indices[0]]),
                'end_px': int(us_tomato[indices[-1]]),
                'center_px': float(us_tomato[indices[len(indices)//2]]),
                'width_px': len(indices),
                'width_mm': len(indices) * params.pixel_size_mm,  # Convert to mm
                'max_depth_mm': float(defect_depths.max()),
                'mean_depth_mm': float(defect_depths.mean()),
                'indices': indices
            })
    
    # Sort by maximum depth
    defects.sort(key=lambda d: -d['max_depth_mm'])
    
    # Re-number IDs after sorting
    for i, d in enumerate(defects, 1):
        d['id'] = i
    
    print(f"   Detected: {len(defects)} defects after filtering")
    
    return defects, defect_mask_full, reference, edge_mask_full, degree_name


# =========================
# VISUALIZATION
# =========================

def visualize(img, us, vs, depths, tomato_mask, defects, defect_mask, 
              reference, edge_mask, degree_name, params):
    """Create comprehensive visualization."""
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Prepare display coordinates
    vs_display = vs.copy()
    if params.flip_image_vertical:
        vs_display = img.shape[0] - 1 - vs_display
    
    valid = ~np.isnan(depths)
    
    # ============ Plot 1: Image with defects ============
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Plot stripe
    ax1.plot(us[tomato_mask], vs_display[tomato_mask], 'lime', linewidth=3, alpha=0.8, label='Tomato')
    
    # Label defects
    used_positions = []
    for d in defects:
        mask = defect_mask & (us >= d['start_px']) & (us <= d['end_px'])
        if not mask.any():
            continue
        
        us_def = us[mask]
        vs_def = vs_display[mask]
        v_min = vs_def.min()
        
        # Color by depth severity
        if d['max_depth_mm'] > 10.0:
            color = '#FF0000'
        elif d['max_depth_mm'] > 5.0:
            color = '#FFA500'
        else:
            color = '#FFD700'
        
        defect_center_u = d['center_px']
        
        # Find non-overlapping label position
        label_v = v_min - 300
        min_spacing = 300
        
        for prev_u, prev_v in used_positions:
            if abs(defect_center_u - prev_u) < 200 and abs(label_v - prev_v) < min_spacing:
                if label_v > img.shape[0] / 2:
                    label_v = prev_v - min_spacing
                else:
                    label_v = prev_v + min_spacing
        
        label_v = max(50, min(label_v, img.shape[0] - 50))
        used_positions.append((defect_center_u, label_v))
        
        # Draw label with connecting line
        ax1.plot([defect_center_u, defect_center_u], 
                [label_v + 100, v_min - 5],
                color=color, linewidth=2, alpha=0.7, zorder=4)
        
        ax1.text(defect_center_u, label_v, f"{d['id']}) D = {d['mean_depth_mm']:.1f} mm, W = {d['width_mm']:.1f} mm", 
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=color, 
                         edgecolor='white', linewidth=3, alpha=0.95),
                zorder=6)
    
    ax1.set_title(f"Detected Defects: {len(defects)}", 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.axis('off')
    
    # ============ Plot 2: Depth Profile ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    us_valid = us[valid]
    depths_valid = depths[valid]
    tomato_valid = tomato_mask[valid]
    edge_valid = edge_mask[valid] if edge_mask is not None else np.zeros(valid.sum(), dtype=bool)
    
    # Convert pixels to mm for X-axis
    us_mm = pixels_to_mm(us_valid, params.cx, params.pixel_size_mm)
    us_mm_all = pixels_to_mm(us, params.cx, params.pixel_size_mm)
    
    # Plot all measured depths (after lowpass + outlier removal)
    ax2.plot(us_mm[tomato_valid], depths_valid[tomato_valid], 
            'blue', linewidth=2, label='Measured (tomato)', zorder=2)
    
    # Plot floor region
    floor_valid = valid & ~tomato_mask
    if floor_valid.any():
        us_floor_mm = pixels_to_mm(us[floor_valid], params.cx, params.pixel_size_mm)
        ax2.plot(us_floor_mm, depths[floor_valid],
                'gray', linewidth=1, alpha=0.3, label='Floor', zorder=1)
    
    # Determine reference method and appropriate label BEFORE plotting
    if edge_mask is not None and edge_mask.any():
        # Check if using defect-adjacent (edge points in center) vs standard edges
        edge_indices = np.where(edge_mask[valid] & tomato_valid)[0]
        tomato_indices = np.where(tomato_valid)[0]
        
        if len(edge_indices) > 0 and len(tomato_indices) > 0:
            # Calculate position of edge points relative to tomato region
            tomato_start = tomato_indices[0]
            tomato_end = tomato_indices[-1]
            tomato_length = tomato_end - tomato_start
            
            # Check if edge points are mostly in center (defect-adjacent) or at edges (standard)
            edge_in_center_count = 0
            for idx in edge_indices:
                relative_pos = (idx - tomato_start) / tomato_length
                if 0.2 < relative_pos < 0.8:  # In center 60%
                    edge_in_center_count += 1
            
            if edge_in_center_count > len(edge_indices) * 0.5:
                # Defect-adjacent reference
                if degree_name == "circular arc":
                    title_text = f'Circular arc fit to defect-adjacent regions'
                else:
                    title_text = f'Polynomial ({degree_name}) fit to defect-adjacent regions'
                ref_label = 'Local reference (near defects)'
            else:
                # Standard edge-based reference
                if degree_name == "circular arc":
                    title_text = f'Circular arc fit to tomato edges'
                else:
                    title_text = f'Polynomial ({degree_name}) fit to tomato edges'
                ref_label = 'Edge regions (tomato ends)'
        else:
            if degree_name == "circular arc":
                title_text = f'Circular arc fit'
            else:
                title_text = f'Polynomial ({degree_name}) fit'
            ref_label = 'Reference points'
    else:
        if degree_name == "circular arc":
            title_text = f'Circular arc fit'
        else:
            title_text = f'Polynomial ({degree_name}) fit'
        ref_label = 'Reference points'
    
    # Highlight edge regions used for reference fitting (plot once with correct label)
    if edge_valid.any():
        ax2.scatter(us_mm[edge_valid & tomato_valid], depths_valid[edge_valid & tomato_valid],
                   c='green', s=30, alpha=0.7, marker='s', label=ref_label, zorder=3)
    
    # Plot reference surface
    ref_valid = reference[valid]
    ref_mask = ~np.isnan(ref_valid)
    ax2.plot(us_mm[ref_mask], ref_valid[ref_mask],
            'orange', linewidth=3, linestyle='--', 
            label='Reference', zorder=4)
    
    # Mark defects with smaller markers
    for d in defects:
        mask = defect_mask[valid] & (us_valid >= d['start_px']) & (us_valid <= d['end_px'])
        if mask.any():
            color = '#FF0000' if d['max_depth_mm'] > 10.0 else '#FFA500' if d['max_depth_mm'] > 5.0 else '#FFD700'
            ax2.scatter(us_mm[mask], depths_valid[mask], c=color, s=20, 
                       marker='v', edgecolors='darkred', linewidths=0.5, zorder=0.5)
    
    ax2.set_xlabel('Position (mm)', fontsize=11)
    ax2.set_ylabel('Depth Z (mm)', fontsize=11)
    ax2.set_title(title_text, fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Zoom in to tomato region only (exclude floor)
    if tomato_valid.any():
        tomato_depths = depths_valid[tomato_valid]
        depth_min = tomato_depths.min()
        depth_max = tomato_depths.max()
        depth_range = depth_max - depth_min
        
        # Add 20% margin above and 15% below for better visibility
        margin_top = depth_range * 0.20
        margin_bottom = depth_range * 0.15
        y_min = depth_min - margin_top
        y_max = depth_max + margin_bottom
        
        ax2.set_ylim(y_max, y_min)  # Inverted: max at bottom, min at top
    else:
        ax2.invert_yaxis()
    
    # ============ Plot 3: Deviation Analysis ============
    ax3 = fig.add_subplot(gs[1, 1])
    
    if tomato_mask.any():
        dev_full = depths - reference
        dev_tomato = dev_full[tomato_mask]
        us_tomato = us[tomato_mask]
        
        valid_dev = ~np.isnan(dev_tomato)
        if valid_dev.any():
            # Plot deviation
            ax3.plot(us_tomato[valid_dev], dev_tomato[valid_dev], 
                    'blue', linewidth=2, label='Deviation')
            ax3.axhline(0, color='green', linewidth=2, alpha=0.6, label='Baseline')
            ax3.axhline(params.defect_threshold_mm, color='red', linewidth=2, 
                       linestyle='--', label=f'Threshold ({params.defect_threshold_mm}mm)')
            
            # Mark edges
            if edge_mask is not None:
                edge_tomato = edge_mask[tomato_mask] & valid_dev
                ax3.scatter(us_tomato[edge_tomato], dev_tomato[edge_tomato],
                          c='green', s=40, alpha=0.5, marker='s', zorder=5)
            
            # Fill defect regions
            for d in defects:
                mask = (us_tomato >= d['start_px']) & (us_tomato <= d['end_px']) & valid_dev
                if mask.any():
                    color = '#FF0000' if d['max_depth_mm'] > 2.0 else '#FFA500' if d['max_depth_mm'] > 1.0 else '#FFD700'
                    ax3.fill_between(us_tomato[mask], 0, dev_tomato[mask],
                                    color=color, alpha=0.3)
    
    ax3.set_xlabel('Image column (pixels)', fontsize=11)
    ax3.set_ylabel('Deviation (mm)', fontsize=11)
    ax3.set_title('Deviation Analysis', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =========================
# RESULTS EXPORT
# =========================

def save_results(us, vs, depths, tomato_mask, reference, defects, defect_mask, params):
    """Save detailed results to CSV."""
    print(f"\nSaving results...")
    
    # Create defect ID array
    defect_ids = np.zeros(len(us), dtype=int)
    tomato_indices = np.where(tomato_mask)[0]
    
    for defect in defects:
        for local_idx in defect['indices']:
            global_idx = tomato_indices[local_idx]
            if defect_mask[global_idx]:
                defect_ids[global_idx] = defect['id']
    
    # Save point cloud with defect labels
    with open(params.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['u_px', 'v_px', 'depth_mm', 'is_tomato', 'reference_mm', 
                        'deviation_mm', 'is_defect', 'defect_id'])
        
        for i in range(len(us)):
            dev = depths[i] - reference[i] if not np.isnan(reference[i]) else np.nan
            writer.writerow([
                us[i], vs[i], depths[i],
                int(tomato_mask[i]),
                reference[i] if not np.isnan(reference[i]) else '',
                dev if not np.isnan(dev) else '',
                int(defect_mask[i]),
                defect_ids[i] if defect_ids[i] > 0 else ''
            ])
    
    # Save defect summary
    summary_file = params.out_csv.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("DEFECT DETECTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total defects: {len(defects)}\n\n")
        
        if defects:
            f.write(f"{'ID':<4} {'Center(px)':<12} {'Width(px)':<10} {'Width(mm)':<10} "
                   f"{'Max Depth(mm)':<14} {'Mean Depth(mm)':<14}\n")
            f.write("-"*70 + "\n")
            
            for d in defects:
                f.write(f"{d['id']:<4} {d['center_px']:>10.1f}   "
                       f"{d['width_px']:>8}   {d['width_mm']:>8.1f}   "
                       f"{d['max_depth_mm']:>12.2f}   "
                       f"{d['mean_depth_mm']:>12.2f}\n")
    
    print(f"Saved: {params.out_csv}")
    print(f"Saved: {summary_file}")


# =========================
# MAIN
# =========================

def main():
    print("\n" + "="*70)
    print("  TOMATO DEFECT DETECTION - IMPROVED VERSION")
    print("  With Enhanced Outlier Removal & Lowpass Filtering")
    print("="*70)
    
    # Load image
    print(f"\nLoading: {P.img_path}")
    img = cv2.imread(P.img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {P.img_path}")
    print(f"Loaded: {img.shape[1]}x{img.shape[0]}")
    
    # Undistort if needed
    img_u, K = undistort(img, P.K, P.dist)
    P.K = K

    # Extract laser stripe
    print(f"\nExtracting {P.laser_color} laser stripe...")
    us, vs = extract_laser_color_ratio(img_u, P)
    
    if len(us) == 0:
        raise RuntimeError("No laser stripe detected!")
    
    print(f"Detected: {len(us)} raw points")
    
    # ============================================================
    # PREPROCESSING PIPELINE (NEW!)
    # ============================================================
    
    # Stage 1: Initial lowpass filter (removes high-frequency noise)
    if P.enable_initial_lowpass:
        us, vs = initial_lowpass_filter(us, vs, 
                                        method=P.lowpass_method,
                                        window=P.lowpass_window)
    
    # Stage 2: Outlier removal (removes anomalous points)
    if P.enable_outlier_removal:
        us, vs = remove_outliers_from_stripe(us, vs, method=P.outlier_method)
    
    # Stage 3: Final smoothing (gentle polish)
    if P.stripe_smoothing_sigma > 0:
        us, vs = smooth_stripe_positions(us, vs, sigma=P.stripe_smoothing_sigma)
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Final clean points: {len(us)}")
    
    # ============================================================
    # Continue with normal pipeline
    # ============================================================
    
    # Triangulate
    print(f"\nTriangulating 3D points...")
    depths = triangulate(us, vs, P)
    valid = ~np.isnan(depths)
    print(f"Valid 3D points: {valid.sum()}")
    
    # Extract valid points
    us_valid = us[valid]
    vs_valid = vs[valid]
    depths_valid = depths[valid]
    
    # Detrend floor tilt
    depths_detrended, floor_trend, floor_baseline = detrend_floor_tilt(
        us_valid, depths_valid, method="linear"
    )
    
    # UPDATE: Put detrended depths back into main array
    depths[valid] = depths_detrended
    depths_valid = depths_detrended
    
    # Segment tomato from floor (using detrended depths)
    tomato_mask_valid, floor_level = segment_tomato(depths_valid, P)
    
    # Map back to full array
    tomato_mask = np.zeros(len(us), dtype=bool)
    tomato_mask[valid] = tomato_mask_valid
    
    print(f"Tomato points: {tomato_mask.sum()}")
    
    if tomato_mask.sum() < P.min_tomato_points:
        print(f"WARNING: Only {tomato_mask.sum()} tomato points (minimum: {P.min_tomato_points})")
    
    # Detect defects
    defects, defect_mask, reference, edge_mask, degree_name = detect_defects(
        us, depths, tomato_mask, P
    )
    
    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS: {len(defects)} DEFECTS DETECTED")
    print(f"{'='*70}")
    
    if defects:
        print(f"\n{'ID':<4} {'Center':<12} {'Width':<18} {'Max Depth':<12} {'Mean Depth':<12}")
        print("-" * 70)
        for d in defects:
            marker = '●' if d['max_depth_mm'] > 1.0 else '○'
            print(f"{marker} {d['id']:<2} {d['center_px']:>8.0f} px  "
                  f"{d['width_px']:>4} px ({d['width_mm']:>5.1f}mm)  "
                  f"{d['max_depth_mm']:>6.2f} mm    "
                  f"{d['mean_depth_mm']:>6.2f} mm")
    else:
        print("\nNO DEFECTS DETECTED - HEALTHY TOMATO!")
    
    # Save results (uncomment if needed)
    # save_results(us, vs, depths, tomato_mask, reference, defects, defect_mask, P)
    
    # Visualize
    visualize(img_u, us, vs, depths, tomato_mask, defects, defect_mask, 
              reference, edge_mask, degree_name, P)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()