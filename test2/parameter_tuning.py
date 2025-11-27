"""
PARAMETER TUNING TOOL - MODULAR VERSION
========================================
Tool to find optimal parameters for laser extraction and segmentation.
Uses the modular tomato defect detection system.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import from modular system
from preprocess import Params, pixels_to_mm, undistort, crop_image
from laser_extraction import extract_laser_color_ratio, preprocess_stripe
from triangulate import triangulate, detrend_floor_tilt
from segmentation import auto_calculate_tomato_gap, segment_tomato


# =========================
# TUNING CONFIGURATION
# =========================

class TuningParams(Params):
    """Extended Params for tuning with additional options."""
    
    def __init__(self, **kwargs):
        # Set defaults then override with kwargs
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Recompute derived values
        self.__post_init__()


def create_tuning_params(
    img_path: str = None,
    laser_color: str = "blue",
    color_ratio_threshold: float = 0.5,
    min_val_fraction: float = 0.25,
    bandpass_kernel: int = 9,
    correction_factor_depth: float = 2.105,
    correction_factor_width: float = 0.25,
    enable_crop: bool = True
) -> Params:
    
    P = Params()
    
    if img_path:
        P.img_path = img_path
    
    P.laser_color = laser_color
    P.color_ratio_threshold = color_ratio_threshold
    P.min_val_fraction = min_val_fraction
    P.bandpass_kernel = bandpass_kernel
    P.correction_factor_depth = correction_factor_depth
    P.correction_factor_width = correction_factor_width
    
    # Cropping
    P.enable_crop = enable_crop
    
    # Recompute derived values
    P.__post_init__()
    
    return P


# =========================
# LASER EXTRACTION TUNING
# =========================

def tune_laser_extraction(
    img_path: str = None,
    laser_color: str = "blue",
    color_ratio_threshold: float = 0.5,
    min_val_fraction: float = 0.25,
    bandpass_kernel: int = 9,
    enable_crop: bool = True
) -> dict:
    
    # Create params
    P = create_tuning_params(
        img_path=img_path,
        laser_color=laser_color,
        color_ratio_threshold=color_ratio_threshold,
        min_val_fraction=min_val_fraction,
        bandpass_kernel=bandpass_kernel,
        enable_crop=enable_crop
    )
    
    # Load image
    print(f"\n{'='*70}")
    print(f"LASER EXTRACTION ANALYSIS")
    print(f"{'='*70}")
    print(f"Image: {P.img_path}")
    
    img = cv2.imread(P.img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {P.img_path}")
    
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")
    
    # Undistort
    img, P.K = undistort(img, P.K, P.dist)
    
    # Crop if enabled
    if P.enable_crop:
        img, P = crop_image(img, P)
    
    print(f"Laser color: {laser_color.upper()}")
    print(f"\nTest parameters:")
    print(f"  color_ratio_threshold: {color_ratio_threshold:.2f}")
    print(f"  min_val_fraction:      {min_val_fraction:.2f}")
    print(f"  bandpass_kernel:       {bandpass_kernel}")
    
    # Extract laser
    us, vs = extract_laser_color_ratio(img, P)
    
    # Calculate mask for statistics
    blue = img[:,:,0].astype(np.float32)
    green = img[:,:,1].astype(np.float32)
    red = img[:,:,2].astype(np.float32)
    
    if laser_color == "blue":
        target = blue
        color_ratio = blue / (red + green + 1e-6)
    elif laser_color == "green":
        target = green
        color_ratio = (blue + green) / (0.5 * red + 1e-6)
    elif laser_color == "red":
        target = red
        color_ratio = red / (blue + green + 1e-6)
    
    brightness_threshold = target.max() * min_val_fraction
    mask = (target >= brightness_threshold) & (color_ratio >= color_ratio_threshold)
    
    mask_percent = 100 * mask.sum() / mask.size
    
    print(f"\nResults:")
    print(f"  Target channel max: {target.max():.0f}")
    print(f"  Color ratio max:    {color_ratio.max():.2f}")
    print(f"  Mask coverage:      {mask_percent:.2f}%")
    print(f"  Stripe points:      {len(us)}")
    
    # Analyze and recommend
    recommended = {
        'laser_color': laser_color,
        'color_ratio_threshold': color_ratio_threshold,
        'min_val_fraction': min_val_fraction,
        'bandpass_kernel': bandpass_kernel,
    }
    
    if len(us) == 0:
        print(f"\nWARNING: No stripe detected")
        recommended['color_ratio_threshold'] = max(0.2, color_ratio_threshold - 0.05)
        recommended['min_val_fraction'] = max(0.15, min_val_fraction - 0.05)
        print(f"  Recommendation: Decrease thresholds")
    elif len(us) < 500:
        print(f"\nWARNING: Very few points")
        recommended['color_ratio_threshold'] = max(0.2, color_ratio_threshold - 0.03)
        recommended['min_val_fraction'] = max(0.15, min_val_fraction - 0.03)
        print(f"  Recommendation: Slightly decrease thresholds")
    elif mask_percent > 10:
        print(f"\nWARNING: Mask coverage too high (noise)")
        recommended['color_ratio_threshold'] = min(0.8, color_ratio_threshold + 0.05)
        recommended['min_val_fraction'] = min(0.4, min_val_fraction + 0.05)
        print(f"  Recommendation: Increase thresholds")
    else:
        print(f"\nStatus: Good extraction")
    
    print(f"\nRECOMMENDED PARAMETERS:")
    print(f"  laser_color:           \"{recommended['laser_color']}\"")
    print(f"  color_ratio_threshold: {recommended['color_ratio_threshold']:.2f}")
    print(f"  min_val_fraction:      {recommended['min_val_fraction']:.2f}")
    print(f"  bandpass_kernel:       {recommended['bandpass_kernel']}")
    print(f"{'='*70}\n")
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if len(us) > 0:
        vs_display = vs.copy()
        if P.flip_image_vertical:
            vs_display = img.shape[0] - 1 - vs_display
        ax.scatter(us, vs_display, c="#00FF00", s=2, alpha=0.8)
    
    ax.set_title(f'Detected Laser Stripe: {len(us)} points | {laser_color.upper()}', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return recommended


# =========================
# SEGMENTATION TUNING
# =========================

def tune_segmentation(
    img_path: str = None,
    laser_color: str = "blue",
    color_ratio_threshold: float = 0.5,
    min_val_fraction: float = 0.25,
    bandpass_kernel: int = 9,
    correction_factor_depth: float = 2.105,
    test_gap_mm: float = None,
    enable_crop: bool = True
) -> float:
    
    # Create params
    P = create_tuning_params(
        img_path=img_path,
        laser_color=laser_color,
        color_ratio_threshold=color_ratio_threshold,
        min_val_fraction=min_val_fraction,
        bandpass_kernel=bandpass_kernel,
        correction_factor_depth=correction_factor_depth,
        enable_crop=enable_crop
    )
    
    print(f"\n{'='*70}")
    print(f"SEGMENTATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Image: {P.img_path}")
    
    # Load and preprocess image
    img = cv2.imread(P.img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {P.img_path}")
    
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")
    
    img, P.K = undistort(img, P.K, P.dist)
    
    if P.enable_crop:
        img, P = crop_image(img, P)
    
    print(f"\nCalibration:")
    print(f"  Correction factor (depth): {P.correction_factor_depth:.3f}")
    print(f"  plane_d: {P.plane_d:.1f}")
    
    # Extract laser stripe
    print(f"\nExtracting {laser_color} laser stripe...")
    us, vs = extract_laser_color_ratio(img, P)
    
    if len(us) == 0:
        print(f"ERROR: No laser stripe detected")
        return None
    
    print(f"  Extracted {len(us)} points")
    
    # Preprocess stripe
    us, vs = preprocess_stripe(us, vs, P)
    
    # Triangulate
    print(f"\nTriangulating...")
    depths = triangulate(us, vs, P)
    valid = ~np.isnan(depths)
    
    if valid.sum() == 0:
        print(f"ERROR: No valid 3D points")
        return None
    
    depths_valid = depths[valid]
    us_valid = us[valid]
    print(f"  {valid.sum()} valid points")
    
    # Detrend floor tilt
    print(f"\nDetrending floor tilt...")
    depths_detrended, trend, floor_median = detrend_floor_tilt(us_valid, depths_valid)
    
    tilt_range = trend.max() - trend.min()
    print(f"  Floor tilt: {tilt_range:.1f}mm")
    print(f"  Floor level: {floor_median:.1f}mm")
    
    # Analyze depth statistics
    n = len(depths_detrended)
    depth_min = depths_detrended.min()
    depth_max = depths_detrended.max()
    depth_range = depth_max - depth_min
    
    print(f"\nDepth statistics:")
    print(f"  Min depth (tomato):  {depth_min:.1f}mm")
    print(f"  Max depth (floor):   {depth_max:.1f}mm")
    print(f"  Range:               {depth_range:.1f}mm")
    print(f"  Floor level:         {floor_median:.1f}mm")
    
    # Auto-calculate gap if not provided
    if test_gap_mm is None:
        test_gap_mm = auto_calculate_tomato_gap(depths_detrended, verbose=False)
        print(f"\nAuto-calculated gap: {test_gap_mm:.1f}mm")
    else:
        print(f"\nTest gap: {test_gap_mm:.1f}mm")
    
    # Test segmentation
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths_detrended[:edge_size], depths_detrended[-edge_size:]]))
    
    tomato_threshold = floor_level - test_gap_mm
    tomato_mask = depths_detrended < tomato_threshold
    
    tomato_count = tomato_mask.sum()
    tomato_percent = 100 * tomato_count / n
    
    print(f"\nSegmentation:")
    print(f"  Floor level:      {floor_level:.1f}mm")
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
        recommended_gap = depth_range * 0.18
        print(f"  Consider decreasing gap to {recommended_gap:.1f}mm")
    elif tomato_percent > 60:
        print(f"\nNote: High tomato coverage")
        recommended_gap = depth_range * 0.28
        print(f"  Consider increasing gap to {recommended_gap:.1f}mm")
    else:
        print(f"\nStatus: Good segmentation")
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED VALUES:")
    print(f"{'='*70}")
    print(f"tomato_floor_gap_mm = {recommended_gap:.1f}")
    print(f"auto_calculate_gap = True  # Or use fixed value above")
    print(f"\nGood range: {depth_range*0.15:.1f} - {depth_range*0.30:.1f}mm")
    print(f"{'='*70}\n")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Detrended profile
    ax1 = axes[0]
    ax1.plot(us_valid, depths_detrended, 'b-', linewidth=1, alpha=0.7, label='Measured')
    ax1.axhline(floor_level, color='red', linewidth=2, linestyle='-', 
                label=f'Floor: {floor_level:.1f}mm')
    ax1.axhline(tomato_threshold, color='green', linewidth=2, linestyle='--',
                label=f'Threshold: {tomato_threshold:.1f}mm')
    ax1.fill_between(us_valid, depth_min, tomato_threshold, alpha=0.2, color='green')
    ax1.set_xlabel('Image column (pixels)')
    ax1.set_ylabel('Depth Z (mm)')
    ax1.set_title(f'Depth Profile (Gap={test_gap_mm:.1f}mm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Segmentation result
    ax2 = axes[1]
    ax2.plot(us_valid[~tomato_mask], depths_detrended[~tomato_mask], 
             'r.', markersize=2, alpha=0.5, label='Floor')
    ax2.plot(us_valid[tomato_mask], depths_detrended[tomato_mask], 
             'g.', markersize=2, alpha=0.7, label='Tomato')
    ax2.axhline(floor_level, color='red', linewidth=2, linestyle='--', alpha=0.5)
    ax2.axhline(tomato_threshold, color='green', linewidth=2, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Image column (pixels)')
    ax2.set_ylabel('Depth Z (mm)')
    ax2.set_title(f'Segmentation: {tomato_count} tomato ({tomato_percent:.1f}%), {n-tomato_count} floor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return recommended_gap

# =========================
# MAIN
# =========================

def main():
    tune_laser_extraction(laser_color='green')
    # tune_segmentation(correction_factor_depth=2.5)

if __name__ == "__main__":
    main()