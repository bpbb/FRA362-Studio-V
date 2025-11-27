"""
PARAMETER TUNING TOOL - CLEAN VERSION
======================================
Simplified tool to find optimal parameters for laser extraction and segmentation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, minimum_filter
from dataclasses import dataclass, field

# =========================
# CONFIGURATION
# =========================
@dataclass
class TuningConfig:
    """Configuration for parameter tuning."""
    
    # Image to analyze
    img_path: str = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\test_photo\red\7M309893.JPG"
    
    # Camera intrinsics
    fx: float = 1450.0
    fy: float = 1450.0
    cx: float = 960.0
    cy: float = 540.0
    flip_image_vertical: bool = True
    
    # Calibration
    camera_angle_from_laser_deg: float = 30.0
    cam_to_object_distance_mm: float = 200.0
    use_scale_correction: bool = True
    correction_factor: float = 2
    
    # Computed
    K: np.ndarray = field(init=False)
    plane_n: np.ndarray = field(init=False)
    plane_d: float = field(init=False)
    
    def __post_init__(self):
        self.K = np.array([[self.fx, 0, self.cx], 
                          [0, self.fy, self.cy], 
                          [0, 0, 1]], dtype=np.float64)
        theta = np.deg2rad(self.camera_angle_from_laser_deg)
        self.plane_n = np.array([0, np.cos(theta), np.sin(theta)])
        self.plane_n /= np.linalg.norm(self.plane_n)
        self.plane_d = -self.cam_to_object_distance_mm * np.sin(theta)
        
        if self.use_scale_correction:
            self.plane_d *= self.correction_factor


CONFIG = TuningConfig()


# =========================
# UTILITIES
# =========================

def extract_laser_with_params(img_bgr, laser_color, color_ratio_threshold, 
                               min_val_fraction, bandpass_kernel, 
                               subpixel_halfwidth=3, flip_vertical=True):
    """Extract laser stripe with given parameters."""
    
    blue = img_bgr[:,:,0].astype(np.float32)
    green = img_bgr[:,:,1].astype(np.float32)
    red = img_bgr[:,:,2].astype(np.float32)
    
    # Apply Gaussian blur
    if bandpass_kernel > 1:
        blue = cv2.GaussianBlur(blue, (bandpass_kernel, bandpass_kernel), 0)
        green = cv2.GaussianBlur(green, (bandpass_kernel, bandpass_kernel), 0)
        red = cv2.GaussianBlur(red, (bandpass_kernel, bandpass_kernel), 0)
    
    # Select target channel and compute ratio
    if laser_color == "blue":
        target = blue
        color_ratio = blue / (red + green + 1e-6)
    elif laser_color == "green":
        target = green
        color_ratio = (blue + green) / (0.5 * red + 1e-6)
    elif laser_color == "red":
        target = red
        color_ratio = red / (blue + green + 1e-6)
    else:
        raise ValueError(f"Unknown laser color: {laser_color}")
    
    # Two-stage thresholding
    brightness_threshold = target.max() * min_val_fraction
    ratio_threshold = color_ratio_threshold
    
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
        
        # Subpixel refinement
        v0 = max(0, peak_v - subpixel_halfwidth)
        v1 = min(H-1, peak_v + subpixel_halfwidth)
        w = col[v0:v1+1].clip(min=0.0)
        if w.sum() <= 0:
            continue
        
        vv = rows[v0:v1+1]
        v_c = (vv * w).sum() / w.sum()
        
        if flip_vertical:
            v_c = H - 1 - v_c
        
        u_list.append(float(u))
        v_list.append(float(v_c))
    
    return np.array(u_list), np.array(v_list), target, color_ratio, mask


def triangulate_simple(us, vs, config):
    """Simple triangulation."""
    ones = np.ones_like(us)
    uv1 = np.stack([us, vs, ones], axis=1)
    
    Kinv = np.linalg.inv(config.K)
    rays = (Kinv @ uv1.T).T
    rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12)
    
    depths = []
    for ray in rays:
        denom = np.dot(config.plane_n, ray)
        if abs(denom) > 1e-9:
            t = -config.plane_d / denom
            if t > 0:
                depths.append(ray[2] * t)
                continue
        depths.append(np.nan)
    
    return np.array(depths)


def detrend_floor(us, depths):
    """Remove floor tilt."""
    n = len(depths)
    edge_size = max(50, int(n * 0.1))
    
    floor_indices = np.concatenate([
        np.arange(edge_size),
        np.arange(n - edge_size, n)
    ])
    floor_us = us[floor_indices]
    floor_depths = depths[floor_indices]
    
    # Fit linear trend
    coeffs = np.polyfit(floor_us, floor_depths, deg=1)
    trend = np.polyval(coeffs, us)
    floor_median = np.median(floor_depths)
    depths_detrended = depths - trend + floor_median
    
    return depths_detrended, trend, floor_median


# =========================
# LASER EXTRACTION TUNING
# =========================

def tune_laser_extraction(img_path=None, laser_color="blue", 
                         test_params=None, save_recommendation=False):
    """
    Tune laser extraction parameters.
    
    Returns:
        dict: Recommended parameters
    """
    
    if img_path is None:
        img_path = CONFIG.img_path
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    
    # Default parameters
    if test_params is None:
        test_params = {
            'color_ratio_threshold': 0.5,
            'min_val_fraction': 0.25,
            'bandpass_kernel': 9,
            'subpixel_halfwidth': 3
        }
    
    # Extract with test parameters
    us, vs, target, color_ratio, mask = extract_laser_with_params(
        img, laser_color,
        test_params['color_ratio_threshold'],
        test_params['min_val_fraction'],
        test_params['bandpass_kernel'],
        test_params.get('subpixel_halfwidth', 3),
        CONFIG.flip_image_vertical
    )
    
    # Statistics
    total_pixels = mask.size
    mask_percent = 100 * mask.sum() / total_pixels
    
    print(f"\n{'='*70}")
    print(f"LASER EXTRACTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Image: {img_path}")
    print(f"Laser color: {laser_color.upper()}")
    print(f"\nTest parameters:")
    print(f"  color_ratio_threshold: {test_params['color_ratio_threshold']:.2f}")
    print(f"  min_val_fraction:      {test_params['min_val_fraction']:.2f}")
    print(f"  bandpass_kernel:       {test_params['bandpass_kernel']}")
    print(f"\nResults:")
    print(f"  Target channel max: {target.max():.0f}")
    print(f"  Color ratio max:    {color_ratio.max():.2f}")
    print(f"  Mask coverage:      {mask_percent:.2f}%")
    print(f"  Stripe points:      {len(us)}")
    
    # Analyze and recommend
    recommended = test_params.copy()
    recommended['laser_color'] = laser_color
    
    if len(us) == 0:
        print(f"\nWARNING: No stripe detected")
        recommended['color_ratio_threshold'] = max(0.2, test_params['color_ratio_threshold'] - 0.01)
        recommended['min_val_fraction'] = max(0.15, test_params['min_val_fraction'] - 0.01)
        print(f"  Recommendation: Decrease thresholds")
    elif len(us) < 5000:
        print(f"\nWARNING: Very few points")
        recommended['color_ratio_threshold'] = max(0.2, test_params['color_ratio_threshold'] - 0.01)
        recommended['min_val_fraction'] = max(0.15, test_params['min_val_fraction'] - 0.01)
        print(f"  Recommendation: Slightly decrease thresholds")
    elif mask_percent > 10:
        print(f"\nWARNING: Mask coverage too high")
        recommended['color_ratio_threshold'] = min(0.8, test_params['color_ratio_threshold'] + 0.01)
        recommended['min_val_fraction'] = min(0.4, test_params['min_val_fraction'] + 0.01)
        print(f"  Recommendation: Increase thresholds")
    else:
        print(f"\nStatus: Good extraction")
    
    print(f"\nRECOMMENDED PARAMETERS:")
    print(f"  color_ratio_threshold: {recommended['color_ratio_threshold']:.2f}")
    print(f"  min_val_fraction:      {recommended['min_val_fraction']:.2f}")
    print(f"  bandpass_kernel:       {recommended['bandpass_kernel']}")
    print(f"{'='*70}\n")
    
    # Simple visualization - just detection result
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(us) > 0:
        vs_display = vs.copy()
        if CONFIG.flip_image_vertical:
            vs_display = img.shape[0] - 1 - vs_display
        ax.scatter(us, vs_display, c="#CEFF48", s=2, alpha=0.8)
    
    ax.set_title(f'Detected Laser Stripe: {len(us)} points | {laser_color.upper()}', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Save recommendation
    if save_recommendation:
        save_path = f"recommended_params_{laser_color}_extraction.txt"
        with open(save_path, 'w') as f:
            f.write(f"# Recommended Laser Extraction Parameters\n")
            f.write(f"# Image: {img_path}\n")
            f.write(f"# Laser color: {laser_color}\n\n")
            f.write(f"laser_color: str = \"{laser_color}\"\n")
            f.write(f"color_ratio_threshold: float = {recommended['color_ratio_threshold']:.2f}\n")
            f.write(f"min_val_fraction: float = {recommended['min_val_fraction']:.2f}\n")
            f.write(f"bandpass_kernel: int = {recommended['bandpass_kernel']}\n")
            f.write(f"subpixel_halfwidth: int = {recommended.get('subpixel_halfwidth', 3)}\n")
        print(f"Saved to: {save_path}\n")
    
    return recommended


# =========================
# SEGMENTATION TUNING
# =========================

def tune_segmentation(img_path=None, laser_params=None, test_gap_mm=None, 
                     correction_factor=None, save_recommendation=False):
    """
    Tune segmentation parameters.
    
    Parameters:
        img_path: Path to test image
        laser_params: Dict with laser extraction params
        test_gap_mm: Test gap value in mm (auto-calculated if None)
        correction_factor: Scale correction factor (uses CONFIG if None)
        save_recommendation: If True, saves recommended params
    
    Returns:
        float: Recommended gap_mm value
    """
    
    if img_path is None:
        img_path = CONFIG.img_path
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    
    # Use provided correction factor or get from CONFIG
    if correction_factor is None:
        correction_factor = CONFIG.correction_factor
        use_correction = CONFIG.use_scale_correction
    else:
        use_correction = True
    
    # Default laser parameters
    if laser_params is None:
        laser_params = {
            'laser_color': 'blue',
            'color_ratio_threshold': 0.5,
            'min_val_fraction': 0.25,
            'bandpass_kernel': 9,
            'subpixel_halfwidth': 3
        }
    
    print(f"\n{'='*70}")
    print(f"SEGMENTATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Image: {img_path}")
    print(f"\nCalibration:")
    print(f"  Correction factor: {correction_factor:.1f}")
    print(f"  Use correction: {use_correction}")
    
    # Extract laser stripe
    print(f"\nExtracting laser stripe...")
    us, vs, _, _, _ = extract_laser_with_params(
        img,
        laser_params['laser_color'],
        laser_params['color_ratio_threshold'],
        laser_params['min_val_fraction'],
        laser_params['bandpass_kernel'],
        laser_params.get('subpixel_halfwidth', 3),
        CONFIG.flip_image_vertical
    )
    
    if len(us) == 0:
        print(f"ERROR: No laser stripe detected")
        return None
    
    print(f"  Extracted {len(us)} points")
    
    # Create temporary config with correct correction factor
    # Only copy fields that should be passed to __init__
    init_fields = {
        'img_path': CONFIG.img_path,
        'fx': CONFIG.fx,
        'fy': CONFIG.fy,
        'cx': CONFIG.cx,
        'cy': CONFIG.cy,
        'flip_image_vertical': CONFIG.flip_image_vertical,
        'camera_angle_from_laser_deg': CONFIG.camera_angle_from_laser_deg,
        'cam_to_object_distance_mm': CONFIG.cam_to_object_distance_mm,
        'use_scale_correction': use_correction,
        'correction_factor': correction_factor
    }
    
    temp_config = TuningConfig(**init_fields)
    
    # Triangulate
    print(f"Triangulating...")
    depths = triangulate_simple(us, vs, temp_config)
    valid = ~np.isnan(depths)
    
    if valid.sum() == 0:
        print(f"ERROR: No valid 3D points")
        return None
    
    depths_valid = depths[valid]
    us_valid = us[valid]
    print(f"  {valid.sum()} valid points")
    
    # Detrend floor tilt
    print(f"Detrending floor tilt...")
    depths_detrended, trend, floor_median = detrend_floor(us_valid, depths_valid)
    
    tilt_range = trend.max() - trend.min()
    print(f"  Floor tilt: {tilt_range:.1f}mm")
    print(f"  Floor level: {floor_median:.1f}mm")
    
    # Analyze depth statistics
    n = len(depths_detrended)
    depth_min = depths_detrended.min()
    depth_max = depths_detrended.max()
    depth_range = depth_max - depth_min
    
    print(f"\nDepth statistics (with correction factor = {correction_factor:.1f}):")
    print(f"  Min depth (tomato):  {depth_min:.1f}mm")
    print(f"  Max depth (floor):   {depth_max:.1f}mm")
    print(f"  Range:               {depth_range:.1f}mm")
    print(f"  Floor level:         {floor_median:.1f}mm")
    
    # Calculate physical values (if correction applied)
    if use_correction:
        physical_min = depth_min / correction_factor
        physical_max = depth_max / correction_factor
        physical_range = depth_range / correction_factor
        physical_floor = floor_median / correction_factor
        
        print(f"\nPhysical measurements (without correction):")
        print(f"  Min depth (tomato):  {physical_min:.1f}mm")
        print(f"  Max depth (floor):   {physical_max:.1f}mm")
        print(f"  Range:               {physical_range:.1f}mm")
        print(f"  Floor level:         {physical_floor:.1f}mm")
    
    # Auto-calculate gap if not provided
    if test_gap_mm is None:
        test_gap_mm = depth_range * 0.2
        print(f"\nAuto-calculated gap: {test_gap_mm:.1f}mm (20% of corrected range)")
    else:
        print(f"\nTest gap: {test_gap_mm:.1f}mm")
    
    # Test segmentation
    tomato_threshold = floor_median - test_gap_mm
    tomato_mask = depths_detrended < tomato_threshold
    
    tomato_count = tomato_mask.sum()
    tomato_percent = 100 * tomato_count / n
    
    print(f"\nSegmentation:")
    print(f"  Tomato threshold: {tomato_threshold:.1f}mm")
    print(f"  Tomato points:    {tomato_count} ({tomato_percent:.1f}%)")
    print(f"  Floor points:     {n - tomato_count} ({100 - tomato_percent:.1f}%)")
    
    # Recommendations
    recommended_gap = test_gap_mm
    
    if tomato_percent < 10:
        print(f"\nWARNING: Too few tomato points")
        recommended_gap = depth_range * 0.15
        print(f"  Recommendation: Decrease gap to {recommended_gap:.1f}mm")
    elif tomato_percent > 80:
        print(f"\nWARNING: Too many tomato points")
        recommended_gap = depth_range * 0.35
        print(f"  Recommendation: Increase gap to {recommended_gap:.1f}mm")
    elif tomato_percent < 20:
        print(f"\nNote: Low tomato coverage")
        recommended_gap = depth_range * 0.17
        print(f"  Consider decreasing gap to {recommended_gap:.1f}mm")
    elif tomato_percent > 60:
        print(f"\nNote: High tomato coverage")
        recommended_gap = depth_range * 0.25
        print(f"  Consider increasing gap to {recommended_gap:.1f}mm")
    else:
        print(f"\nStatus: Good segmentation")
    
    print(f"\n{'='*70}")
    print(f"VALUE FOR YOUR MAIN SCRIPT:")
    print(f"{'='*70}")
    print(f"tomato_floor_gap_mm: float = {recommended_gap:.1f}")
    print(f"{'='*70}")
    
    print(f"\nAdditional info:")
    print(f"  Good range: {depth_range*0.15:.1f} - {depth_range*0.30:.1f}mm")
    print(f"  Current measured gap: {floor_median - depth_min:.1f}mm")
    
    if use_correction:
        physical_gap = recommended_gap / correction_factor
        print(f"  Physical gap (before correction): {physical_gap:.1f}mm")
    
    print(f"{'='*70}\n")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Detrended profile
    ax1.plot(us_valid, depths_detrended, 'b-', linewidth=1, alpha=0.7, label='Measured')
    ax1.axhline(floor_median, color='red', linewidth=2, linestyle='-', 
                label=f'Floor: {floor_median:.1f}mm')
    ax1.axhline(tomato_threshold, color='green', linewidth=2, linestyle='--',
                label=f'Threshold: {tomato_threshold:.1f}mm')
    ax1.fill_between(us_valid, depth_min, tomato_threshold,
                     alpha=0.2, color='green')
    ax1.set_xlabel('Image column (pixels)')
    ax1.set_ylabel('Depth Z (mm)')
    title_str = f'Depth Profile (Gap={test_gap_mm:.1f}mm'
    if use_correction:
        title_str += f', CF={correction_factor:.1f}'
    title_str += ')'
    ax1.set_title(title_str)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Segmentation result
    ax2.plot(us_valid[~tomato_mask], depths_detrended[~tomato_mask], 
             'r.', markersize=2, alpha=0.5, label='Floor')
    ax2.plot(us_valid[tomato_mask], depths_detrended[tomato_mask], 
             'g.', markersize=2, alpha=0.7, label='Tomato')
    ax2.axhline(floor_median, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax2.axhline(tomato_threshold, color='green', linewidth=2, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Image column (pixels)')
    ax2.set_ylabel('Depth Z (mm)')
    ax2.set_title(f'Segmentation: {tomato_count} tomato, {n-tomato_count} floor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # Save recommendation
    if save_recommendation:
        save_path = "recommended_params_segmentation.txt"
        with open(save_path, 'w') as f:
            f.write(f"# Recommended Segmentation Parameters\n")
            f.write(f"# Image: {img_path}\n")
            f.write(f"# Correction factor: {correction_factor:.1f}\n")
            f.write(f"# Depth range (corrected): {depth_range:.1f}mm\n")
            if use_correction:
                f.write(f"# Depth range (physical): {physical_range:.1f}mm\n")
            f.write(f"\n")
            f.write(f"# COPY-PASTE THIS VALUE:\n")
            f.write(f"tomato_floor_gap_mm: float = {recommended_gap:.1f}\n")
            f.write(f"\n")
            f.write(f"min_tomato_points: int = 50\n")
            f.write(f"segmentation_method: str = \"curvature\"\n\n")
            f.write(f"# Good gap range: {depth_range*0.15:.1f} - {depth_range*0.30:.1f}mm\n")
            if use_correction:
                f.write(f"# Physical gap (before correction): {physical_gap:.1f}mm\n")
        print(f"Saved to: {save_path}\n")
    
    return recommended_gap


# =========================
# COMPLETE WORKFLOW
# =========================

def complete_tuning_workflow(img_path=None, laser_color="blue"):
    """
    Complete tuning workflow: laser extraction + segmentation.
    """
    
    if img_path is None:
        img_path = CONFIG.img_path
    
    print(f"\n{'#'*70}")
    print(f"COMPLETE PARAMETER TUNING")
    print(f"{'#'*70}")
    print(f"Image: {img_path}")
    print(f"Laser: {laser_color.upper()}")
    print(f"{'#'*70}\n")
    
    # Step 1: Tune laser extraction
    print(f"STEP 1: LASER EXTRACTION")
    print(f"-" * 70)
    
    laser_params = tune_laser_extraction(
        img_path=img_path,
        laser_color=laser_color,
        save_recommendation=True
    )
    
    input("Press Enter to continue to segmentation...")
    
    # Step 2: Tune segmentation
    print(f"\nSTEP 2: SEGMENTATION")
    print(f"-" * 70)
    
    gap_mm = tune_segmentation(
        img_path=img_path,
        laser_params=laser_params,
        save_recommendation=True
    )
    
    # Summary
    print(f"\n{'#'*70}")
    print(f"TUNING COMPLETE")
    print(f"{'#'*70}")
    print(f"\nRecommended parameters:")
    print(f"\n# Laser Extraction:")
    print(f"laser_color = \"{laser_color}\"")
    print(f"color_ratio_threshold = {laser_params['color_ratio_threshold']:.2f}")
    print(f"min_val_fraction = {laser_params['min_val_fraction']:.2f}")
    print(f"bandpass_kernel = {laser_params['bandpass_kernel']}")
    print(f"\n# Segmentation:")
    print(f"tomato_floor_gap_mm = {gap_mm:.1f}")
    print(f"\nSaved to files:")
    print(f"  - recommended_params_{laser_color}_extraction.txt")
    print(f"  - recommended_params_segmentation.txt")
    print(f"{'#'*70}\n")

# =========================
# MAIN
# =========================

def main():
    # complete_tuning_workflow(laser_color='blue')
    tune_laser_extraction(laser_color='red')
    # tune_segmentation(correction_factor=3.2)


if __name__ == "__main__":
    main()