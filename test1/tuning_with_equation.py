import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter1d, minimum_filter, median_filter, label as ndimage_label
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
import csv
import copy

# =========================
# CALIBRATION PARAMETERS
# =========================
@dataclass
class CalibrationImage:
    """Represents one calibration image with known defect size"""
    path: str
    actual_defect_width_mm: float  # Known actual width in mm
    actual_defect_depth_mm: float  # Known actual depth in mm (optional, use 0 if unknown)
    laser_color: str = ""
    description: str = ""

# =========================
# DEFINE YOUR CALIBRATION IMAGES HERE
# =========================
CALIBRATION_IMAGES = [
    CalibrationImage(
        path=r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\test_photo\blue\7M309820.jpg",
        actual_defect_width_mm=5.0,  # REPLACE with actual measured width
        actual_defect_depth_mm=4.0,  # REPLACE with actual measured depth (or 0 if unknown)
        laser_color="blue",
        description="Small defect"
    ),
    CalibrationImage(
        path=r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\test_photo\blue\7M309824.jpg",
        actual_defect_width_mm=10.0,  # REPLACE with actual measured width
        actual_defect_depth_mm=6.0,   # REPLACE with actual measured depth (or 0 if unknown)
        laser_color="blue",
        description="Medium defect"
    ),
    CalibrationImage(
        path=r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\test_photo\blue\7M309828.jpg",
        actual_defect_width_mm=10.0,  # REPLACE with actual measured width
        actual_defect_depth_mm=8.0,   # REPLACE with actual measured depth (or 0 if unknown)
        laser_color="blue",
        description="Large defect"
    ),
    CalibrationImage(
        path=r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\test_photo\blue\7M309832.jpg",
        actual_defect_width_mm=20.0,   # REPLACE with actual measured width
        actual_defect_depth_mm=10.0,   # REPLACE with actual measured depth (or 0 if unknown)
        laser_color="blue",
        description="Test defect"
    ),
]

# =========================
# BASE PARAMETERS (same as main file)
# =========================
@dataclass
class Params:
    # Calibration
    camera_angle_from_laser_deg: float = 30.0
    cam_to_object_distance_mm: float = 200.0
    
    # Scale correction - WILL BE TUNED
    use_scale_correction: bool = True
    correction_factor_width: float = 1.0
    correction_factor_depth: float = 1.0
    
    # Camera intrinsics
    fx: float = 1450.0
    fy: float = 1450.0
    cx: float = 960.0
    cy: float = 540.0
    dist: np.ndarray = field(default_factory=lambda: np.array([0,0,0,0,0], dtype=np.float64))
    flip_image_vertical: bool = True
    
    # Laser Extraction
    laser_color: str = "red"
    bandpass_kernel: int = 9
    subpixel_halfwidth: int = 3
    color_ratio_threshold: float = 0.5
    min_val_fraction: float = 0.25
    
    # Preprocessing
    enable_initial_lowpass: bool = True
    lowpass_method: str = "savgol"
    lowpass_window: int = 11
    
    enable_outlier_removal: bool = True
    outlier_method: str = "improved"
    
    stripe_smoothing_sigma: float = 2.0
    
    # Segmentation
    auto_calculate_gap: bool = True
    tomato_floor_gap_mm: float = 35.0
    min_tomato_points: int = 80
    
    # Edge-Poly Reference
    edge_region_percent: float = 25.0
    edge_poly_degree: int = 2
    use_circular_reference: bool = True
    defect_adjacent_points: int = 100
    
    # Defect Detection
    defect_threshold_mm: float = 1.0  # Lower for calibration - adjust if needed
    min_defect_width_px: int = 5      # Smaller for calibration
    edge_exclude_percent: float = 5.0 # Less exclusion at edges
    
    # Smoothing
    depth_smoothing_sigma: float = 5.0
    use_median_prefilter: bool = True
    median_window_size: int = 5
    
    # Clustering
    cluster_defects: bool = True
    cluster_max_gap_pixels: int = 10
    
    # Computed
    K: np.ndarray = field(init=False)
    plane_n: np.ndarray = field(init=False)
    plane_d: float = field(init=False)
    pixel_size_mm: float = field(init=False)
    
    def __post_init__(self):
        self.K = np.array([[self.fx, 0, self.cx], 
                           [0, self.fy, self.cy], 
                           [0, 0, 1]], dtype=np.float64)
        self._calculate_derived_values()

    def _calculate_derived_values(self):
        """Recalculate plane_d and pixel_size_mm based on current factors."""
        theta = np.deg2rad(self.camera_angle_from_laser_deg)
        
        # Calculate plane normal vector
        self.plane_n = np.array([0, np.cos(theta), np.sin(theta)])
        self.plane_n /= np.linalg.norm(self.plane_n)
        
        # Calculate base plane distance
        base_d = -self.cam_to_object_distance_mm * np.sin(theta)
        
        # plane_d is used in triangulation (for depth) -> use depth factor
        if self.use_scale_correction:
            self.plane_d = base_d * self.correction_factor_depth
        else:
            self.plane_d = base_d
            
        avg_z = abs(self.plane_d)
        
        # pixel_size_mm is used for defect width calculation -> use width factor
        if self.use_scale_correction:
             # Calculate base pixel size first, then apply width correction
            base_pixel_size = avg_z / self.fx
            self.pixel_size_mm = base_pixel_size * self.correction_factor_width
        else:
            self.pixel_size_mm = avg_z / self.fx
    
    def update_correction_factors(self, new_factor_width, new_factor_depth):
            """Update both correction factors and recalculate derived values"""
            self.correction_factor_width = new_factor_width
            self.correction_factor_depth = new_factor_depth
            self._calculate_derived_values()


# =========================
# IMPORT FUNCTIONS FROM MAIN FILE
# =========================

def undistort(image, K, dist):
    """Undistort image using camera calibration."""
    if np.allclose(dist, 0):
        return image, K
    h, w = image.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
    und = cv2.undistort(image, K, dist, None, newK)
    return und, newK


def extract_laser_color_ratio(img_bgr, params):
    """Extract laser using color ratio."""
    blue = img_bgr[:,:,0].astype(np.float32)
    green = img_bgr[:,:,1].astype(np.float32)
    red = img_bgr[:,:,2].astype(np.float32)
    
    if params.bandpass_kernel > 1:
        blue = cv2.GaussianBlur(blue, (params.bandpass_kernel, params.bandpass_kernel), 0)
        green = cv2.GaussianBlur(green, (params.bandpass_kernel, params.bandpass_kernel), 0)
        red = cv2.GaussianBlur(red, (params.bandpass_kernel, params.bandpass_kernel), 0)
    
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
        raise ValueError(f"Unknown laser color: {params.laser_color}")
    
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
        
        peak_v = int(idxs[np.argmax(col[idxs])])
        
        v0 = max(0, peak_v - params.subpixel_halfwidth)
        v1 = min(H-1, peak_v + params.subpixel_halfwidth)
        w = col[v0:v1+1].clip(min=0.0)
        if w.sum() <= 0:
            continue
        
        vv = rows[v0:v1+1]
        v_c = (vv * w).sum() / w.sum()
        
        if params.flip_image_vertical:
            v_c = H - 1 - v_c
        
        u_list.append(float(u))
        v_list.append(float(v_c))
    
    return np.array(u_list), np.array(v_list)


def initial_lowpass_filter(us, vs, method="savgol", window=11):
    """Apply initial lowpass filter to raw stripe data."""
    if len(us) < window:
        return us, vs
    
    vs_original = vs.copy()
    
    if method == "savgol":
        polyorder = min(3, window - 2)
        if window % 2 == 0:
            window += 1
        vs_filtered = savgol_filter(vs, window, polyorder, mode='nearest')
    elif method == "gaussian":
        sigma = window / 6.0
        vs_filtered = gaussian_filter1d(vs, sigma=sigma, mode='nearest')
    elif method == "moving_avg":
        from scipy.ndimage import uniform_filter1d
        vs_filtered = uniform_filter1d(vs, size=window, mode='nearest')
    else:
        raise ValueError(f"Unknown filter method: {method}")
    
    return us, vs_filtered


def remove_outliers_from_stripe(us, vs, method="improved"):
    """Remove outliers from extracted laser stripe."""
    if len(us) < 10:
        return us, vs
    
    initial_count = len(us)
    
    if method == "improved":
        # Stage 2.1: Large jumps
        dv = np.diff(vs)
        dv_pad = np.concatenate([[0], dv])
        mad = np.median(np.abs(dv - np.median(dv)))
        threshold_jump = 5.0 * (mad * 1.4826)
        mask_continuous = np.abs(dv_pad) < threshold_jump
        us = us[mask_continuous]
        vs = vs[mask_continuous]
        
        if len(us) < 10:
            return us, vs
        
        # Stage 2.2: Local MAD
        window = 31
        half_window = window // 2
        outlier_mask = np.ones(len(vs), dtype=bool)
        
        for i in range(len(vs)):
            start = max(0, i - half_window)
            end = min(len(vs), i + half_window + 1)
            local_window = vs[start:end]
            local_median = np.median(local_window)
            local_mad = np.median(np.abs(local_window - local_median))
            
            if local_mad < 0.1:
                threshold = 2.0
            else:
                threshold = 3.5 * (local_mad * 1.4826)
            
            if np.abs(vs[i] - local_median) > threshold:
                outlier_mask[i] = False
        
        us = us[outlier_mask]
        vs = vs[outlier_mask]
        
        if len(us) < 10:
            return us, vs
        
        # Stage 2.3: Isolated points
        gaps = np.diff(us)
        gap_threshold = np.median(gaps) + 3 * np.std(gaps)
        isolated_mask = np.ones(len(us), dtype=bool)
        for i in range(1, len(us) - 1):
            if gaps[i-1] > gap_threshold and gaps[i] > gap_threshold:
                isolated_mask[i] = False
        
        us = us[isolated_mask]
        vs = vs[isolated_mask]
    
    return us, vs


def smooth_stripe_positions(us, vs, sigma=2.0):
    """Apply final smoothing to stripe positions."""
    if sigma <= 0 or len(us) < 5:
        return us, vs
    vs_smooth = gaussian_filter1d(vs, sigma=sigma, mode='nearest')
    return us, vs_smooth


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


def detrend_floor_tilt(us, depths):
    """Remove floor tilt."""
    n = len(depths)
    edge_size = max(50, int(n * 0.1))
    
    floor_indices = np.concatenate([
        np.arange(edge_size),
        np.arange(n - edge_size, n)
    ])
    floor_us = us[floor_indices]
    floor_depths = depths[floor_indices]
    
    coeffs = np.polyfit(floor_us, floor_depths, deg=1)
    trend = np.polyval(coeffs, us)
    floor_median = np.median(floor_depths)
    depths_detrended = depths - trend + floor_median
    
    return depths_detrended, trend, floor_median


def auto_calculate_tomato_gap(depths):
    """Auto-calculate tomato gap."""
    n = len(depths)
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths[:edge_size], depths[-edge_size:]]))
    
    depth_range = depths.max() - depths.min()
    initial_gap = depth_range * 0.20
    
    tomato_threshold = floor_level - initial_gap
    tomato_mask_test = depths < tomato_threshold
    tomato_percent = 100 * tomato_mask_test.sum() / n
    
    if tomato_percent < 10:
        recommended_gap = depth_range * 0.18
    elif tomato_percent > 70:
        recommended_gap = depth_range * 0.40
    elif tomato_percent < 25:
        recommended_gap = depth_range * 0.22
    elif tomato_percent > 55:
        recommended_gap = depth_range * 0.32
    else:
        recommended_gap = depth_range * 0.27
    
    return recommended_gap


def segment_tomato(depths, params):
    """Segment tomato from floor."""
    n = len(depths)
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths[:edge_size], depths[-edge_size:]]))
    
    if params.auto_calculate_gap:
        gap_mm = auto_calculate_tomato_gap(depths)
    else:
        gap_mm = params.tomato_floor_gap_mm
    
    tomato_mask = depths < (floor_level - gap_mm)
    
    labeled, num = ndimage_label(tomato_mask)
    
    if num > 0:
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        largest = np.argmax(sizes) + 1
        tomato_mask = (labeled == largest)
        
        if tomato_mask.sum() < params.min_tomato_points:
            tomato_mask = np.zeros(n, dtype=bool)
    else:
        tomato_mask = np.zeros(n, dtype=bool)
    
    return tomato_mask, floor_level


def fit_circular_arc(us, depths):
    """Fit circular arc."""
    from scipy.optimize import least_squares
    
    u_mid = (us.max() + us.min()) / 2
    d_mid = (depths.max() + depths.min()) / 2
    r_init = max(us.max() - us.min(), depths.max() - depths.min()) / 2
    
    def residuals(params):
        u_c, d_c, R = params
        return np.sqrt((us - u_c)**2 + (depths - d_c)**2) - R
    
    result = least_squares(residuals, [u_mid, d_mid, r_init])
    u_c, d_c, R = result.x
    
    u_all = np.arange(us.min(), us.max() + 1)
    under_sqrt = R**2 - (u_all - u_c)**2
    valid = under_sqrt >= 0
    
    d_all = np.full(len(u_all), np.nan)
    
    if valid.any():
        d_upper = d_c + np.sqrt(under_sqrt[valid])
        d_lower = d_c - np.sqrt(under_sqrt[valid])
        
        if depths.mean() > d_c:
            d_all[valid] = d_upper
        else:
            d_all[valid] = d_lower
    
    return u_all, d_all, (u_c, d_c, R)


def create_defect_adjacent_reference(us, depths, params, rough_defect_mask=None):
    """Create reference surface."""
    n = len(us)
    
    if rough_defect_mask is not None and rough_defect_mask.any():
        edge_mask = np.zeros(n, dtype=bool)
        reference = np.full(n, np.nan)
        
        labeled, num_defects = ndimage_label(rough_defect_mask)
        
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
            
            base_margin = params.defect_adjacent_points
            if defect_width < 20:
                edge_margin = max(15, base_margin // 2)
            elif defect_width < 50:
                edge_margin = base_margin
            else:
                edge_margin = int(base_margin * 1.5)
            
            left_start = max(0, start_idx - edge_margin)
            left_end = start_idx
            left_indices = np.arange(left_start, left_end)
            
            right_start = end_idx + 1
            right_end = min(n, end_idx + 1 + edge_margin)
            right_indices = np.arange(right_start, right_end)
            
            defect_mean_depth = depths[defect_indices].mean()
            
            valid_left = False
            valid_right = False
            
            if len(left_indices) > 5:
                left_mean_depth = depths[left_indices].mean()
                left_std = depths[left_indices].std()
                if left_mean_depth < defect_mean_depth - 0.5 and left_std < 3.0:
                    valid_left = True
            
            if len(right_indices) > 5:
                right_mean_depth = depths[right_indices].mean()
                right_std = depths[right_indices].std()
                if right_mean_depth < defect_mean_depth - 0.5 and right_std < 3.0:
                    valid_right = True
            
            if not (valid_left or valid_right):
                continue
            
            if valid_left and len(left_indices) > 0:
                edge_mask[left_indices] = True
                total_edge_points += len(left_indices)
            
            if valid_right and len(right_indices) > 0:
                edge_mask[right_indices] = True
                total_edge_points += len(right_indices)
            
            if valid_left and valid_right:
                edge_indices = np.concatenate([left_indices, right_indices])
                edge_us = us[edge_indices]
                edge_depths = depths[edge_indices]
                
                if len(edge_us) >= params.edge_poly_degree + 1:
                    coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                    local_region = np.arange(left_start, right_end)
                    reference[local_region] = np.polyval(coeffs, us[local_region])
                    valid_defects += 1
                else:
                    left_mean = depths[left_indices].mean()
                    right_mean = depths[right_indices].mean()
                    interp_region = np.arange(start_idx, end_idx + 1)
                    t = (us[interp_region] - us[start_idx]) / (us[end_idx] - us[start_idx] + 1e-9)
                    reference[interp_region] = left_mean + t * (right_mean - left_mean)
                    valid_defects += 1
            elif valid_left:
                left_mean = depths[left_indices].mean()
                local_region = np.arange(left_start, end_idx + 1)
                reference[local_region] = left_mean
                valid_defects += 1
            elif valid_right:
                right_mean = depths[right_indices].mean()
                local_region = np.arange(start_idx, right_end)
                reference[local_region] = right_mean
                valid_defects += 1
        
        degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
            params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
        )
        
        if total_edge_points < 20 or valid_defects == 0:
            edge_size = max(5, int(n * params.edge_region_percent / 100.0))
            edge_mask = np.zeros(n, dtype=bool)
            edge_mask[:edge_size] = True
            edge_mask[-edge_size:] = True
            
            edge_us = us[edge_mask]
            edge_depths = depths[edge_mask]
            
            if params.use_circular_reference and len(edge_us) >= 10:
                try:
                    u_arc, d_arc, _ = fit_circular_arc(edge_us, edge_depths)
                    reference = np.interp(us, u_arc, d_arc, left=np.nan, right=np.nan)
                    degree_name = "circular arc"
                except:
                    coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                    reference = np.polyval(coeffs, us)
            else:
                coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                reference = np.polyval(coeffs, us)
    else:
        edge_size = max(5, int(n * params.edge_region_percent / 100.0))
        edge_mask = np.zeros(n, dtype=bool)
        edge_mask[:edge_size] = True
        edge_mask[-edge_size:] = True
        
        edge_us = us[edge_mask]
        edge_depths = depths[edge_mask]
        
        if params.use_circular_reference and len(edge_us) >= 10:
            try:
                u_arc, d_arc, _ = fit_circular_arc(edge_us, edge_depths)
                reference = np.interp(us, u_arc, d_arc, left=np.nan, right=np.nan)
                degree_name = "circular arc"
            except Exception:
                coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
                reference = np.polyval(coeffs, us)
                degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
                    params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
                )
        else:
            coeffs = np.polyfit(edge_us, edge_depths, params.edge_poly_degree)
            reference = np.polyval(coeffs, us)
            degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
                params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
            )
    
    return reference, edge_mask, degree_name


def detect_defects(us, depths, tomato_mask, params):
    """Detect defects with two-pass approach."""
    us_tomato = us[tomato_mask]
    depths_tomato = depths[tomato_mask]
    
    if len(us_tomato) < 10:
        return [], np.zeros(len(us), dtype=bool), np.full(len(us), np.nan), None, None
    
    # Pre-smooth
    if params.depth_smoothing_sigma > 0:
        if params.use_median_prefilter:
            window = int(params.median_window_size)
            if window % 2 == 0:
                window += 1
            depths_clean = median_filter(depths_tomato, size=window, mode='nearest')
        else:
            depths_clean = depths_tomato.copy()
        
        depths_processed = gaussian_filter1d(depths_clean, sigma=params.depth_smoothing_sigma, mode='nearest')
    else:
        depths_processed = depths_tomato.copy()
    
    # Pass 1: Rough detection
    ref_rough, edge_mask_rough, _ = create_defect_adjacent_reference(
        us_tomato, depths_processed, params, rough_defect_mask=None
    )
    
    dev_rough = depths_processed - ref_rough
    
    if params.depth_smoothing_sigma > 0:
        deviation_sigma = params.depth_smoothing_sigma * 0.5
        if deviation_sigma > 0:
            dev_rough = gaussian_filter1d(dev_rough, sigma=deviation_sigma, mode='nearest')
    
    rough_defect_mask = dev_rough > params.defect_threshold_mm
    
    if params.edge_exclude_percent > 0:
        edge_margin = int(len(depths_tomato) * params.edge_exclude_percent / 100.0)
        if edge_margin > 0:
            rough_defect_mask[:edge_margin] = False
            rough_defect_mask[-edge_margin:] = False
    
    labeled_rough, num_rough = ndimage_label(rough_defect_mask)
    
    for i in range(1, num_rough + 1):
        if (labeled_rough == i).sum() < params.min_defect_width_px:
            rough_defect_mask[labeled_rough == i] = False
    
    num_rough_defects = (ndimage_label(rough_defect_mask)[1])
    
    if num_rough_defects == 0:
        ref_tomato = ref_rough
        edge_mask_tomato = edge_mask_rough
        dev_tomato = dev_rough
        defect_mask_tomato = rough_defect_mask
        degree_name = {1: "linear", 2: "quadratic", 3: "cubic"}.get(
            params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
        )
    else:
        # Pass 2: Refined detection
        ref_tomato, edge_mask_tomato, degree_name = create_defect_adjacent_reference(
            us_tomato, depths_processed, params, rough_defect_mask=rough_defect_mask
        )
        
        dev_tomato = depths_processed - ref_tomato
        
        if params.depth_smoothing_sigma > 0:
            deviation_sigma = params.depth_smoothing_sigma * 0.5
            if deviation_sigma > 0:
                dev_tomato = gaussian_filter1d(dev_tomato, sigma=deviation_sigma, mode='nearest')
        
        defect_mask_tomato = dev_tomato > params.defect_threshold_mm
        
        if params.edge_exclude_percent > 0:
            edge_margin = int(len(depths_tomato) * params.edge_exclude_percent / 100.0)
            if edge_margin > 0:
                defect_mask_tomato[:edge_margin] = False
                defect_mask_tomato[-edge_margin:] = False
    
    # Map back
    reference = np.full(len(us), np.nan)
    reference[tomato_mask] = ref_tomato
    
    defect_mask_full = np.zeros(len(us), dtype=bool)
    defect_mask_full[tomato_mask] = defect_mask_tomato
    
    edge_mask_full = np.zeros(len(us), dtype=bool)
    edge_mask_full[tomato_mask] = edge_mask_tomato
    
    # Cluster defects
    defects = []
    
    if params.cluster_defects:
        labeled, num = ndimage_label(defect_mask_tomato)
        
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
                'width_mm': len(indices) * params.pixel_size_mm,
                'max_depth_mm': float(defect_depths.max()),
                'mean_depth_mm': float(defect_depths.mean()),
                'indices': indices
            })
    
    defects.sort(key=lambda d: -d['max_depth_mm'])
    
    for i, d in enumerate(defects, 1):
        d['id'] = i
    
    return defects, defect_mask_full, reference, edge_mask_full, degree_name


# =========================
# CALIBRATION FUNCTIONS
# =========================

def process_single_image_silent(img_path, params, image_laser_color, verbose=False):
    """
    Process a single image and return detected defects.
    Silent version for calibration (no prints unless error).
    
    Returns:
        defects: List of detected defects with measurements
        success: Boolean indicating if processing succeeded
    """
    # Temporarily set the image's laser color
    original_laser_color = params.laser_color
    params.laser_color = image_laser_color
    
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            if verbose:
                print(f"  ERROR: Cannot read {img_path}")
            params.laser_color = original_laser_color
            return [], False
        
        if verbose:
            print(f"  Loaded: {img.shape[1]}x{img.shape[0]}")
        
        # Undistort
        img_u, K = undistort(img, params.K, params.dist)
        params.K = K
        
        # Extract laser
        us, vs = extract_laser_color_ratio(img_u, params)
        if len(us) == 0:
            if verbose:
                print(f"  ERROR: No laser stripe detected")
            params.laser_color = original_laser_color
            return [], False
        
        if verbose:
            print(f"  Laser points: {len(us)}")
        
        # Preprocessing
        if params.enable_initial_lowpass:
            us, vs = initial_lowpass_filter(us, vs, method=params.lowpass_method, window=params.lowpass_window)
        
        if params.enable_outlier_removal:
            us, vs = remove_outliers_from_stripe(us, vs, method=params.outlier_method)
        
        if params.stripe_smoothing_sigma > 0:
            us, vs = smooth_stripe_positions(us, vs, sigma=params.stripe_smoothing_sigma)
        
        if verbose:
            print(f"  After preprocessing: {len(us)} points")
        
        # Triangulate
        depths = triangulate(us, vs, params)
        valid = ~np.isnan(depths)
        
        if valid.sum() < 50:
            if verbose:
                print(f"  ERROR: Too few valid 3D points ({valid.sum()})")
            params.laser_color = original_laser_color
            return [], False
        
        us_valid = us[valid]
        vs_valid = vs[valid]
        depths_valid = depths[valid]
        
        if verbose:
            print(f"  Valid 3D points: {valid.sum()}")
            print(f"  Depth range: {depths_valid.min():.1f} to {depths_valid.max():.1f} mm")
        
        # Detrend
        depths_detrended, _, _ = detrend_floor_tilt(us_valid, depths_valid)
        depths[valid] = depths_detrended
        
        if verbose:
            print(f"  Detrended depth range: {depths_detrended.min():.1f} to {depths_detrended.max():.1f} mm")
        
        # Segment
        tomato_mask_valid, floor_level = segment_tomato(depths_detrended, params)
        tomato_mask = np.zeros(len(us), dtype=bool)
        tomato_mask[valid] = tomato_mask_valid
        
        if tomato_mask.sum() < params.min_tomato_points:
            if verbose:
                print(f"  ERROR: Too few tomato points ({tomato_mask.sum()}, need {params.min_tomato_points})")
            params.laser_color = original_laser_color
            return [], False
        
        if verbose:
            tomato_depths = depths[tomato_mask]
            print(f"  Tomato points: {tomato_mask.sum()}")
            print(f"  Tomato depth range: {tomato_depths.min():.1f} to {tomato_depths.max():.1f} mm")
            print(f"  Defect threshold: {params.defect_threshold_mm:.1f} mm")
        
        # Detect defects
        defects, _, _, _, _ = detect_defects(us, depths, tomato_mask, params)
        
        if verbose:
            print(f"  Defects found: {len(defects)}")
            if len(defects) > 0:
                for d in defects:
                    print(f"    - Width: {d['width_mm']:.1f}mm, Depth: {d['max_depth_mm']:.1f}mm")
        
        params.laser_color = original_laser_color
        return defects, True
        
    except Exception as e:
        if verbose:
            print(f"ERROR processing {img_path}: {e}")
        params.laser_color = original_laser_color
        return [], False


def calculate_error_metrics(calib_images, factor_width, factor_depth, params_template, verbose=False):
    """
    Calculate error metrics for given correction factors across all calibration images.
    """
    # Create params with these correction factors
    params = copy.deepcopy(params_template)
    params.update_correction_factors(factor_width, factor_depth)
    
    width_errors = []
    depth_errors = []
    results = []
    
    for calib_img in calib_images:
        # Pass the image's specific laser color
        defects, success = process_single_image_silent(calib_img.path, params, calib_img.laser_color, verbose=verbose)
        
        if not success or len(defects) == 0:
            width_errors.append(100.0)
            depth_errors.append(100.0)
            results.append({
                'image': calib_img.description,
                'success': False,
                'detected': False
            })
            continue
        
        largest_defect = defects[0]
        
        width_error = abs(largest_defect['width_mm'] - calib_img.actual_defect_width_mm)
        
        if calib_img.actual_defect_depth_mm > 0:
            depth_error = abs(largest_defect['max_depth_mm'] - calib_img.actual_defect_depth_mm)
        else:
            depth_error = 0
            
        width_errors.append(width_error)
        depth_errors.append(depth_error)
        
        results.append({
            'image': calib_img.description,
            'success': True,
            'detected': True,
            'measured_width_mm': largest_defect['width_mm'],
            'actual_width_mm': calib_img.actual_defect_width_mm,
            'width_error_mm': width_error,
            'width_error_percent': 100 * width_error / calib_img.actual_defect_width_mm if calib_img.actual_defect_width_mm > 0 else 0,
            'measured_depth_mm': largest_defect['max_depth_mm'],
            'actual_depth_mm': calib_img.actual_defect_depth_mm,
            'depth_error_mm': depth_error if calib_img.actual_defect_depth_mm > 0 else None,
        })
    
    # Calculate total error (weighted combination)
    mean_width_error = np.mean(width_errors)
    # Filter out 0 errors which means no actual depth was provided
    valid_depth_errors = [e for i, e in enumerate(depth_errors) if calib_images[i].actual_defect_depth_mm > 0 and e < 100]
    mean_depth_error = np.mean(valid_depth_errors) if valid_depth_errors else 0

    # Total error combines both
    total_error = mean_width_error + mean_depth_error # Using simple sum for robustness
    
    return total_error, mean_width_error, mean_depth_error, results

def find_optimal_correction_factor_simultaneous(calib_images, params_template,
                                               search_range=(0.1, 5.0),
                                               initial_guess_width=1.0,
                                               initial_guess_depth=1.0):
    """
    Find optimal correction factors using simultaneous optimization.
    Uses scipy.optimize.minimize with gradient-based methods for better convergence.
    
    This approach optimizes both factors together, potentially finding better
    global optima than sequential optimization.
    """
    from scipy.optimize import minimize, Bounds
    
    print(f"\n{'='*70}")
    print("  SIMULTANEOUS CORRECTION FACTOR CALIBRATION")
    print(f"{'='*70}")
    print(f"\nUsing gradient-based optimization (faster and more accurate)")
    print(f"Initial guesses: width={initial_guess_width:.3f}, depth={initial_guess_depth:.3f}")
    
    iteration_count = [0]
    best_error = [float('inf')]
    best_factors = [initial_guess_width, initial_guess_depth]
    
    def objective(factors):
        """Combined objective function for both width and depth"""
        factor_width, factor_depth = factors
        iteration_count[0] += 1
        
        # Calculate combined error
        total_error, mean_width_error, mean_depth_error, _ = calculate_error_metrics(
            calib_images, factor_width, factor_depth, params_template, verbose=False
        )
        
        # Track best result
        if total_error < best_error[0]:
            best_error[0] = total_error
            best_factors[0] = factor_width
            best_factors[1] = factor_depth
        
        # Progress feedback
        print(f"  Iter {iteration_count[0]}: W={factor_width:.3f}, D={factor_depth:.3f} -> "
              f"W_err={mean_width_error:.2f}mm, D_err={mean_depth_error:.2f}mm, "
              f"Total={total_error:.2f}mm", end='\r')
        
        return total_error
    
    # Set up bounds
    bounds = Bounds([search_range[0], search_range[0]], 
                   [search_range[1], search_range[1]])
    
    # Initial guess
    x0 = np.array([initial_guess_width, initial_guess_depth])
    
    print(f"\nOptimizing both factors simultaneously...")
    
    # Use L-BFGS-B algorithm (gradient-based, handles bounds)
    result = minimize(
        objective,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'ftol': 1e-4,
            'gtol': 1e-4,
            'maxiter': 50,
            'disp': False
        }
    )
    
    optimal_factor_width = result.x[0]
    optimal_factor_depth = result.x[1]
    
    print(f"\n\n{'='*70}")
    print("  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOptimal Width Factor:  {optimal_factor_width:.3f}")
    print(f"Optimal Depth Factor:  {optimal_factor_depth:.3f}")
    print(f"Final error: {result.fun:.2f} mm")
    print(f"Converged: {result.success}")
    print(f"Iterations: {result.nit}")
    
    return optimal_factor_width, optimal_factor_depth, result.fun


def find_optimal_correction_factor_sequential(calib_images, params_template,
                                             search_range=(0.5, 10.0),
                                             initial_guess_width=1.0,
                                             initial_guess_depth=1.0):
    """
    Find the optimal correction factors sequentially: Width first, then Depth.
    """
    print(f"\n{'='*70}")
    print("  SEQUENTIAL CORRECTION FACTOR CALIBRATION")
    print(f"{'='*70}")
    
    current_factor_width = initial_guess_width
    current_factor_depth = initial_guess_depth

    # --- Step 1: Optimize Width Correction Factor ---
    print(f"\n[Step 1/2] Optimizing Width Correction Factor...")
    
    iteration_count = [0]  # Use list to make it mutable in closure
    
    def objective_width(factor_width):
        """Objective function to minimize for width only"""
        iteration_count[0] += 1
        # Calculate error with current depth factor and new width factor
        _, mean_width_error, _, _ = calculate_error_metrics(
            calib_images, factor_width, current_factor_depth, params_template, verbose=False
        )
        # Print progress
        print(f"  Iteration {iteration_count[0]}: width={factor_width:.3f} -> error={mean_width_error:.2f}mm", end='\r')
        # Minimize the mean absolute width error
        return mean_width_error
    
    result_width = minimize_scalar(
        objective_width,
        bounds=search_range,
        method='bounded',
        options={'xatol': 0.001, 'maxiter': 50}
    )
    
    optimal_factor_width = result_width.x
    current_factor_width = optimal_factor_width
    print(f"\n  Width Optimization Complete: {optimal_factor_width:.3f} (Error: {result_width.fun:.2f} mm)")


    # --- Step 2: Optimize Depth Correction Factor ---
    print(f"\n[Step 2/2] Optimizing Depth Correction Factor...")
    
    # Check if there's any depth data to calibrate
    if all(img.actual_defect_depth_mm == 0 for img in calib_images):
        print("  Skipping Depth Optimization: No actual defect depth data provided.")
        optimal_factor_depth = current_factor_depth
    else:
        iteration_count = [0]  # Reset counter
        
        def objective_depth(factor_depth):
            """Objective function to minimize for depth only"""
            iteration_count[0] += 1
            # Calculate error with optimal width factor and new depth factor
            _, _, mean_depth_error, _ = calculate_error_metrics(
                calib_images, current_factor_width, factor_depth, params_template, verbose=False
            )
            # Print progress
            print(f"  Iteration {iteration_count[0]}: depth={factor_depth:.3f} -> error={mean_depth_error:.2f}mm", end='\r')
            # Minimize the mean absolute depth error
            return mean_depth_error
        
        result_depth = minimize_scalar(
            objective_depth,
            bounds=search_range,
            method='bounded',
            options={'xatol': 0.001, 'maxiter': 50}
        )
        
        optimal_factor_depth = result_depth.x
        print(f"\n  Depth Optimization Complete: {optimal_factor_depth:.3f} (Error: {result_depth.fun:.2f} mm)")


    # Final error calculation
    final_error, mean_width_error, mean_depth_error, _ = calculate_error_metrics(
        calib_images, optimal_factor_width, optimal_factor_depth, params_template, verbose=False
    )
    
    print(f"\n{'='*70}")
    print("  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOptimal Width Correction Factor: {optimal_factor_width:.3f}")
    print(f"Optimal Depth Correction Factor: {optimal_factor_depth:.3f}")
    print(f"Final mean width error: {mean_width_error:.2f} mm")
    print(f"Final mean depth error: {mean_depth_error:.2f} mm")
    print(f"Total error metric: {final_error:.2f} mm")
    
    return optimal_factor_width, optimal_factor_depth, final_error

def find_optimal_correction_factor(calib_images, params_template, 
                                   search_range=(0.5, 10.0), 
                                   initial_guess=3.2):
    """
    Find the optimal correction factor using optimization.
    
    Args:
        calib_images: List of CalibrationImage objects
        params_template: Template Params object with all settings
        search_range: Tuple of (min, max) correction factor to search
        initial_guess: Initial guess for correction factor
    
    Returns:
        optimal_factor: Best correction factor found
        optimization_result: Detailed optimization results
    """
    print(f"\n{'='*70}")
    print("  CORRECTION FACTOR CALIBRATION")
    print(f"{'='*70}")
    print(f"\nCalibration images: {len(calib_images)}")
    print(f"Search range: {search_range[0]:.2f} to {search_range[1]:.2f}")
    print(f"Initial guess: {initial_guess:.2f}")
    
    # Define objective function
    def objective(correction_factor):
        """Objective function to minimize"""
        total_error, _, _, _ = calculate_error_metrics(
            calib_images, correction_factor, params_template, verbose=False
        )
        return total_error
    
    print(f"\nOptimizing correction factor...")
    print(f"This may take 1-2 minutes...\n")
    
    # Use scipy's minimize_scalar for 1D optimization
    result = minimize_scalar(
        objective,
        bounds=search_range,
        method='bounded',
        options={'xatol': 0.01, 'maxiter': 50}
    )
    
    optimal_factor = result.x
    optimal_error = result.fun
    
    print(f"\n{'='*70}")
    print("  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOptimal correction factor: {optimal_factor:.3f}")
    print(f"Final error metric: {optimal_error:.2f} mm")
    print(f"Optimization converged: {result.success}")
    print(f"Iterations: {result.nfev}")
    
    return optimal_factor, result


def generate_calibration_equations(calib_images, params_template, optimal_factor_width, optimal_factor_depth):
    """
    Generate polynomial calibration equations that map measured values to actual values.
    
    Returns calibration equations:
        actual_width = f(measured_width)
        actual_depth = f(measured_depth)
    
    These can be used to correct measurements from the system.
    """
    print(f"\nCollecting calibration data points...")
    
    # Get measurements with optimal factors
    params = copy.deepcopy(params_template)
    params.update_correction_factors(optimal_factor_width, optimal_factor_depth)
    
    measured_widths = []
    actual_widths = []
    measured_depths = []
    actual_depths = []
    
    for calib_img in calib_images:
        defects, success = process_single_image_silent(
            calib_img.path, params, calib_img.laser_color, verbose=False
        )
        
        if success and len(defects) > 0:
            measured_widths.append(defects[0]['width_mm'])
            actual_widths.append(calib_img.actual_defect_width_mm)
            
            if calib_img.actual_defect_depth_mm > 0:
                measured_depths.append(defects[0]['max_depth_mm'])
                actual_depths.append(calib_img.actual_defect_depth_mm)
    
    if len(measured_widths) < 2:
        print(f"  WARNING: Not enough data points for calibration equations")
        return None, None
    
    print(f"  Width data points: {len(measured_widths)}")
    print(f"  Depth data points: {len(measured_depths)}")
    
    # Fit polynomial equations
    print(f"\nFitting calibration polynomials...")
    
    # Width calibration equation
    measured_widths = np.array(measured_widths)
    actual_widths = np.array(actual_widths)
    
    # Try different polynomial degrees and pick best
    best_width_degree = 1
    best_width_score = float('inf')
    best_width_coeffs = None
    
    for degree in [1, 2, 3]:
        if len(measured_widths) >= degree + 1:
            coeffs = np.polyfit(measured_widths, actual_widths, degree)
            predicted = np.polyval(coeffs, measured_widths)
            rmse = np.sqrt(np.mean((actual_widths - predicted)**2))
            
            if rmse < best_width_score:
                best_width_score = rmse
                best_width_degree = degree
                best_width_coeffs = coeffs
    
    width_eq = {
        'degree': best_width_degree,
        'coefficients': best_width_coeffs,
        'rmse': best_width_score,
        'measured_range': (measured_widths.min(), measured_widths.max()),
        'r_squared': 1 - (np.sum((actual_widths - np.polyval(best_width_coeffs, measured_widths))**2) / 
                         np.sum((actual_widths - actual_widths.mean())**2))
    }
    
    print(f"\n  Width Calibration Equation (degree {best_width_degree}):")
    print(f"    actual_width = ", end='')
    for i, c in enumerate(best_width_coeffs):
        power = best_width_degree - i
        if i > 0:
            print(f" + " if c >= 0 else f" - ", end='')
            print(f"{abs(c):.6f}", end='')
        else:
            print(f"{c:.6f}", end='')
        
        if power > 0:
            print(f"*measured^{power}" if power > 1 else "*measured", end='')
    print(f"\n    RMSE: {best_width_score:.3f} mm")
    print(f"    R²: {width_eq['r_squared']:.4f}")
    print(f"    Valid range: {width_eq['measured_range'][0]:.1f} - {width_eq['measured_range'][1]:.1f} mm")
    
    # Depth calibration equation
    depth_eq = None
    if len(measured_depths) >= 2:
        measured_depths = np.array(measured_depths)
        actual_depths = np.array(actual_depths)
        
        best_depth_degree = 1
        best_depth_score = float('inf')
        best_depth_coeffs = None
        
        for degree in [1, 2, 3]:
            if len(measured_depths) >= degree + 1:
                coeffs = np.polyfit(measured_depths, actual_depths, degree)
                predicted = np.polyval(coeffs, measured_depths)
                rmse = np.sqrt(np.mean((actual_depths - predicted)**2))
                
                if rmse < best_depth_score:
                    best_depth_score = rmse
                    best_depth_degree = degree
                    best_depth_coeffs = coeffs
        
        depth_eq = {
            'degree': best_depth_degree,
            'coefficients': best_depth_coeffs,
            'rmse': best_depth_score,
            'measured_range': (measured_depths.min(), measured_depths.max()),
            'r_squared': 1 - (np.sum((actual_depths - np.polyval(best_depth_coeffs, measured_depths))**2) / 
                             np.sum((actual_depths - actual_depths.mean())**2))
        }
        
        print(f"\n  Depth Calibration Equation (degree {best_depth_degree}):")
        print(f"    actual_depth = ", end='')
        for i, c in enumerate(best_depth_coeffs):
            power = best_depth_degree - i
            if i > 0:
                print(f" + " if c >= 0 else f" - ", end='')
                print(f"{abs(c):.6f}", end='')
            else:
                print(f"{c:.6f}", end='')
            
            if power > 0:
                print(f"*measured^{power}" if power > 1 else "*measured", end='')
        print(f"\n    RMSE: {best_depth_score:.3f} mm")
        print(f"    R²: {depth_eq['r_squared']:.4f}")
        print(f"    Valid range: {depth_eq['measured_range'][0]:.1f} - {depth_eq['measured_range'][1]:.1f} mm")
    else:
        print(f"\n  Depth Calibration: Not enough data points")
    
    # Generate Python code for easy copy-paste
    print(f"\n{'='*70}")
    print("CALIBRATION FUNCTIONS")
    print(f"{'='*70}\n")
    
    print("calibrate_width:")
    print(f'coeffs = {list(best_width_coeffs)}')
    print()
    
    if depth_eq:
        print("calibrate_depth:")
        print(f'coeffs = {list(best_depth_coeffs)}')
        print()
    
    return width_eq, depth_eq


def visualize_calibration_results(calib_images, params_template, optimal_factor_width, optimal_factor_depth):
    """
    Visualize calibration results comparing before/after optimization for both width and depth.
    """
    print(f"\n{'='*70}")
    print("  DETAILED CALIBRATION RESULTS")
    print(f"{'='*70}")
    
    params_template_copy = copy.deepcopy(params_template)
    
    original_factor_width = params_template_copy.correction_factor_width
    original_factor_depth = params_template_copy.correction_factor_depth
    
    print(f"\nComparing:")
    print(f"  Original Factors (W/D): {original_factor_width:.3f} / {original_factor_depth:.3f}")
    print(f"  Optimal Factors (W/D):  {optimal_factor_width:.3f} / {optimal_factor_depth:.3f}")
    print(f"  Width Change: {((optimal_factor_width / original_factor_width - 1) * 100):+.1f}%")
    print(f"  Depth Change: {((optimal_factor_depth / original_factor_depth - 1) * 100):+.1f}%")
    
    # Get results for both factors
    _, mean_width_err_orig, mean_depth_err_orig, results_original = calculate_error_metrics(
        calib_images, original_factor_width, original_factor_depth, params_template, verbose=False
    )
    
    _, mean_width_err_opt, mean_depth_err_opt, results_optimal = calculate_error_metrics(
        calib_images, optimal_factor_width, optimal_factor_depth, params_template, verbose=False
    )
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"PER-IMAGE COMPARISON")
    print(f"{'='*70}\n")
    
    for i, calib_img in enumerate(calib_images):
        print(f"{calib_img.description} ({calib_img.laser_color.upper()} Laser):")
        
        # Width comparison
        print(f"  Actual width: {calib_img.actual_defect_width_mm:.1f} mm")
        if results_original[i]['detected']:
            print(f"  Original (W={original_factor_width:.2f}): {results_original[i]['measured_width_mm']:.2f} mm (Err: {results_original[i]['width_error_mm']:+.2f} mm, {results_original[i]['width_error_percent']:+.1f}%)")
        else:
            print(f"  Original (W={original_factor_width:.2f}): NOT DETECTED")
        if results_optimal[i]['detected']:
            print(f"  Optimal  (W={optimal_factor_width:.2f}): {results_optimal[i]['measured_width_mm']:.2f} mm (Err: {results_optimal[i]['width_error_mm']:+.2f} mm, {results_optimal[i]['width_error_percent']:+.1f}%)")
        else:
            print(f"  Optimal  (W={optimal_factor_width:.2f}): NOT DETECTED")
            
        # Depth comparison
        if calib_img.actual_defect_depth_mm > 0:
            print(f"  Actual depth: {calib_img.actual_defect_depth_mm:.1f} mm")
            if results_original[i]['detected']:
                depth_err_orig = results_original[i]['depth_error_mm'] if results_original[i]['depth_error_mm'] is not None else 0
                print(f"  Original (D={original_factor_depth:.2f}): {results_original[i]['measured_depth_mm']:.2f} mm (Err: {depth_err_orig:+.2f} mm)")
            if results_optimal[i]['detected']:
                depth_err_opt = results_optimal[i]['depth_error_mm'] if results_optimal[i]['depth_error_mm'] is not None else 0
                print(f"  Optimal  (D={optimal_factor_depth:.2f}): {results_optimal[i]['measured_depth_mm']:.2f} mm (Err: {depth_err_opt:+.2f} mm)")
        
        print()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print(f"OVERALL IMPROVEMENT")
    print(f"{'='*70}")
    print(f"Width Error:  {mean_width_err_orig:.2f} mm → {mean_width_err_opt:.2f} mm ({((mean_width_err_opt/mean_width_err_orig - 1)*100):+.1f}%)")
    if mean_depth_err_orig > 0:
        print(f"Depth Error:  {mean_depth_err_orig:.2f} mm → {mean_depth_err_opt:.2f} mm ({((mean_depth_err_opt/mean_depth_err_orig - 1)*100):+.1f}%)")




# =========================
# MAIN CALIBRATION
# =========================

def main():
    """Main calibration procedure"""
    
    # Verify calibration images exist
    print(f"\nVerifying calibration images...")
    valid_images = []
    
    for calib_img in CALIBRATION_IMAGES:
        img = cv2.imread(calib_img.path)
        if img is None:
            print(f"  WARNING: Cannot read '{calib_img.path}' - skipping")
        else:
            print(f"  ✓ {calib_img.description}: {img.shape[1]}x{img.shape[0]} ({calib_img.laser_color.upper()} Laser)")
            valid_images.append(calib_img)
    
    if len(valid_images) < 2:
        print(f"\nERROR: Need at least 2 valid calibration images (found {len(valid_images)})")
        print(f"\nPlease update the CALIBRATION_IMAGES list at the top of this file.")
        return
    
    print(f"\nUsing {len(valid_images)} calibration images")
    
    # DIAGNOSTIC: Test first image with verbose output
    print(f"\n{'='*70}")
    print("DIAGNOSTIC TEST - Processing first image with verbose output")
    print(f"{'='*70}")
    params_test = Params()
    test_img = valid_images[0]
    print(f"\nTesting: {test_img.description}")
    print(f"Path: {test_img.path}")
    print(f"Expected: width={test_img.actual_defect_width_mm}mm, depth={test_img.actual_defect_depth_mm}mm")
    print(f"Laser color: {test_img.laser_color}")
    print(f"Defect threshold: {params_test.defect_threshold_mm}mm")
    test_defects, test_success = process_single_image_silent(
        test_img.path, params_test, test_img.laser_color, verbose=True
    )
    
    if not test_success:
        print(f"\n⚠ DIAGNOSTIC FAILED - Cannot process images")
        print(f"Check:")
        print(f"  1. Image paths are correct")
        print(f"  2. Laser color setting matches your images")
        print(f"  3. defect_threshold_mm ({params_test.defect_threshold_mm}mm) isn't too high")
        print(f"  4. Images contain visible laser stripe")
        return
    
    if len(test_defects) == 0:
        print(f"\n⚠ DIAGNOSTIC: Image processed OK but no defects detected")
        print(f"Possible issues:")
        print(f"  1. defect_threshold_mm ({params_test.defect_threshold_mm}mm) is too high")
        print(f"     Try reducing it to 1.0 or 0.5 mm")
        print(f"  2. Defect is too small (min width: {params_test.min_defect_width_px} pixels)")
        print(f"  3. Reference surface is following the defect instead of being flat")
        print(f"\nContinuing anyway to test optimization...")
    else:
        print(f"\n✓ DIAGNOSTIC PASSED - Detected {len(test_defects)} defect(s)")
    
    print(f"\n{'='*70}\n")
    
    # Create parameter template
    params_template = Params()
    
    # Define search range
    search_range = (0.1, 5.0)
    
    print(f"\nStarting calibration with:")
    print(f"  Initial Width Factor: {params_template.correction_factor_width:.3f}")
    print(f"  Initial Depth Factor: {params_template.correction_factor_depth:.3f}")
    print(f"  Search Range: {search_range[0]:.1f} to {search_range[1]:.1f}")
    print(f"\nThis will optimize:")
    print(f"  - Width factor: affects measured defect width in mm")
    print(f"  - Depth factor: affects measured defect depth in mm")
    
    # Select optimization method
    print(f"\n{'='*70}")
    print("SELECT OPTIMIZATION METHOD")
    print(f"{'='*70}")
    print(f"\n1. Sequential (default) - Fast, optimizes width then depth")
    print(f"   Best for: Quick calibration, well-behaved error surfaces")
    print(f"\n2. Simultaneous - Gradient-based, optimizes both together")
    print(f"   Best for: Better accuracy, coupled width-depth relationships")
    
    method_choice = input(f"\nSelect method (1-2): ").strip()
    
    if method_choice == '2':
        print(f"\n→ Using Simultaneous Optimization")
        optimal_factor_width, optimal_factor_depth, final_error = find_optimal_correction_factor_simultaneous(
            valid_images, 
            params_template,
            search_range=search_range,
            initial_guess_width=params_template.correction_factor_width,
            initial_guess_depth=params_template.correction_factor_depth
        )
    else:
        print(f"\n→ Using Sequential Optimization (default)")
        optimal_factor_width, optimal_factor_depth, final_error = find_optimal_correction_factor_sequential(
            valid_images, 
            params_template,
            search_range=search_range,
            initial_guess_width=params_template.correction_factor_width,
            initial_guess_depth=params_template.correction_factor_depth
        )
    
    # Visualize results
    visualize_calibration_results(valid_images, params_template, optimal_factor_width, optimal_factor_depth)
    
    # Generate calibration equations
    print(f"\n{'='*70}")
    print("GENERATING CALIBRATION EQUATIONS")
    print(f"{'='*70}")
    
    width_eq, depth_eq = generate_calibration_equations(
        valid_images, params_template, optimal_factor_width, optimal_factor_depth
    )
    
    print(f"\n{'='*70}")
    print("  CALIBRATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Optimal Width Factor: {optimal_factor_width:.3f}")
    print(f"✓ Optimal Depth Factor: {optimal_factor_depth:.3f}")
    print(f"\nTo use this calibration:")
    print(f"\n1. Update defect_detection.py Params class:")
    print(f"     correction_factor_width: float = {optimal_factor_width:.3f}")
    print(f"     correction_factor_depth: float = {optimal_factor_depth:.3f}")
    print(f"\n2. OR use calibration equations for post-processing:")
    print(f"     from calibration_equations import calibrate_width, calibrate_depth")
    print(f"     actual_width = calibrate_width(measured_width)")
    print(f"     actual_depth = calibrate_depth(measured_depth)")
    print(f"\n   Calibration equations saved to:")
    print(f"     /mnt/user-data/outputs/calibration_equations.py")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()