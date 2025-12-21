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

def create_tuning_params(
    img_path: str,
    laser_color: str = "blue",
    color_ratio_threshold: float = 0.49,
    min_val_fraction: float = 0.24,
    bandpass_kernel: int = 9,
    correction_factor_depth: float = 0.9578,
    correction_factor_width: float = 2.2080,
    use_scale_correction: bool = True,
    enable_crop: bool = True,
    crop_x_start: int = 1700,
    crop_x_end: int = 4300,
    crop_y_start: int = 1000,
    crop_y_end: int = 2900,
    fx: float = 4719.1,
    fy: float = 4705.9,
    cx: float = 3000.0,
    cy: float = 2000.0
) -> Params:
    """Create Params object with specified tuning parameters."""
    
    P = Params()
    
    # Image settings
    P.img_path = img_path
    
    # Laser extraction
    P.laser_color = laser_color
    P.color_ratio_threshold = color_ratio_threshold
    P.min_val_fraction = min_val_fraction
    P.bandpass_kernel = bandpass_kernel
    
    # Scale correction
    P.use_scale_correction = use_scale_correction
    P.correction_factor_depth = correction_factor_depth
    P.correction_factor_width = correction_factor_width
    
    # Cropping
    P.enable_crop = enable_crop
    P.crop_x_start = crop_x_start
    P.crop_x_end = crop_x_end
    P.crop_y_start = crop_y_start
    P.crop_y_end = crop_y_end
    
    # Camera intrinsics
    P.fx = fx
    P.fy = fy
    P.cx = cx
    P.cy = cy
    
    # Recompute derived values
    P._recalculate_derived_params()
    
    return P


# =========================
# LASER EXTRACTION TUNING
# =========================

def tune_laser_extraction(
    img_path: str,
    laser_color: str = "blue",
    color_ratio_threshold: float = 0.49,
    min_val_fraction: float = 0.24,
    bandpass_kernel: int = 9,
    enable_crop: bool = True,
    crop_x_start: int = 1700,
    crop_x_end: int = 4300,
    crop_y_start: int = 1000,
    crop_y_end: int = 2900,
    fx: float = 4719.1,
    fy: float = 4705.9,
    cx: float = 3000.0,
    cy: float = 2000.0
) -> dict:
    """
    Tune laser extraction parameters and visualize results.
    
    Args:
        img_path: Path to test image
        laser_color: 'blue', 'green', or 'red'
        color_ratio_threshold: Threshold for color ratio filtering
        min_val_fraction: Minimum brightness as fraction of max
        bandpass_kernel: Kernel size for bandpass filtering
        enable_crop: Whether to crop the image
        crop_x_start, crop_x_end, crop_y_start, crop_y_end: Crop region
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        dict: Recommended parameters
    """
    
    # Create params
    P = create_tuning_params(
        img_path=img_path,
        laser_color=laser_color,
        color_ratio_threshold=color_ratio_threshold,
        min_val_fraction=min_val_fraction,
        bandpass_kernel=bandpass_kernel,
        enable_crop=enable_crop,
        crop_x_start=crop_x_start,
        crop_x_end=crop_x_end,
        crop_y_start=crop_y_start,
        crop_y_end=crop_y_end,
        fx=fx, fy=fy, cx=cx, cy=cy
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
        print(f"Cropped to: {img.shape[1]}x{img.shape[0]}")
    
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
        print(f"\n⚠️ WARNING: No stripe detected")
        recommended['color_ratio_threshold'] = max(0.2, color_ratio_threshold - 0.05)
        recommended['min_val_fraction'] = max(0.15, min_val_fraction - 0.05)
        print(f"  💡 Recommendation: Decrease thresholds")
    elif len(us) < 500:
        print(f"\n⚠️ WARNING: Very few points")
        recommended['color_ratio_threshold'] = max(0.2, color_ratio_threshold - 0.03)
        recommended['min_val_fraction'] = max(0.15, min_val_fraction - 0.03)
        print(f"  💡 Recommendation: Slightly decrease thresholds")
    elif mask_percent > 10:
        print(f"\n⚠️ WARNING: Mask coverage too high (noise)")
        recommended['color_ratio_threshold'] = min(0.8, color_ratio_threshold + 0.05)
        recommended['min_val_fraction'] = min(0.4, min_val_fraction + 0.05)
        print(f"  💡 Recommendation: Increase thresholds")
    else:
        print(f"\n✅ Status: Good extraction")
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED PARAMETERS:")
    print(f"{'='*70}")
    print(f"laser_color           = \"{recommended['laser_color']}\"")
    print(f"color_ratio_threshold = {recommended['color_ratio_threshold']:.2f}")
    print(f"min_val_fraction      = {recommended['min_val_fraction']:.2f}")
    print(f"bandpass_kernel       = {recommended['bandpass_kernel']}")
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
    img_path: str,
    laser_color: str = "blue",
    color_ratio_threshold: float = 0.49,
    min_val_fraction: float = 0.24,
    bandpass_kernel: int = 9,
    correction_factor_depth: float = 0.9578,
    correction_factor_width: float = 2.2080,
    use_scale_correction: bool = True,
    test_gap_mm: float = None,
    enable_crop: bool = True,
    crop_x_start: int = 1700,
    crop_x_end: int = 4300,
    crop_y_start: int = 1000,
    crop_y_end: int = 2900,
    fx: float = 4719.1,
    fy: float = 4705.9,
    cx: float = 3000.0,
    cy: float = 2000.0
) -> float:
    """
    Tune segmentation parameters and visualize tomato/floor separation.
    
    Args:
        img_path: Path to test image
        laser_color: 'blue', 'green', or 'red'
        color_ratio_threshold: Threshold for color ratio filtering
        min_val_fraction: Minimum brightness as fraction of max
        bandpass_kernel: Kernel size for bandpass filtering
        correction_factor_depth: Scale correction for depth measurements
        correction_factor_width: Scale correction for width measurements
        use_scale_correction: Whether to apply scale corrections
        test_gap_mm: Tomato-floor gap to test (None = auto-calculate)
        enable_crop: Whether to crop the image
        crop_x_start, crop_x_end, crop_y_start, crop_y_end: Crop region
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        float: Recommended gap value in mm
    """
    
    # Create params
    P = create_tuning_params(
        img_path=img_path,
        laser_color=laser_color,
        color_ratio_threshold=color_ratio_threshold,
        min_val_fraction=min_val_fraction,
        bandpass_kernel=bandpass_kernel,
        correction_factor_depth=correction_factor_depth,
        correction_factor_width=correction_factor_width,
        use_scale_correction=use_scale_correction,
        enable_crop=enable_crop,
        crop_x_start=crop_x_start,
        crop_x_end=crop_x_end,
        crop_y_start=crop_y_start,
        crop_y_end=crop_y_end,
        fx=fx, fy=fy, cx=cx, cy=cy
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
        print(f"Cropped to: {img.shape[1]}x{img.shape[0]}")
    
    print(f"\nCalibration:")
    print(f"  Use scale correction:      {P.use_scale_correction}")
    print(f"  Correction factor (depth): {P.correction_factor_depth:.4f}")
    print(f"  Correction factor (width): {P.correction_factor_width:.4f}")
    print(f"  plane_d: {P.plane_d:.1f} mm")
    
    # Extract laser stripe
    print(f"\nExtracting {laser_color} laser stripe...")
    us, vs = extract_laser_color_ratio(img, P)
    
    if len(us) == 0:
        print(f"❌ ERROR: No laser stripe detected")
        return None
    
    print(f"  ✓ Extracted {len(us)} points")
    
    # Preprocess stripe
    us, vs = preprocess_stripe(us, vs, P)
    
    # Triangulate
    print(f"\nTriangulating...")
    depths = triangulate(us, vs, P)
    valid = ~np.isnan(depths)
    
    if valid.sum() == 0:
        print(f"❌ ERROR: No valid 3D points")
        return None
    
    depths_valid = depths[valid]
    us_valid = us[valid]
    print(f"  ✓ {valid.sum()} valid points")
    
    # Detrend floor tilt
    print(f"\nDetrending floor tilt...")
    depths_detrended, trend, floor_median = detrend_floor_tilt(us_valid, depths_valid)
    
    tilt_range = trend.max() - trend.min()
    print(f"  Floor tilt: {tilt_range:.1f} mm")
    print(f"  Floor level: {floor_median:.1f} mm")
    
    # Analyze depth statistics
    n = len(depths_detrended)
    depth_min = depths_detrended.min()
    depth_max = depths_detrended.max()
    depth_range = depth_max - depth_min
    
    print(f"\nDepth statistics:")
    print(f"  Min depth (tomato):  {depth_min:.1f} mm")
    print(f"  Max depth (floor):   {depth_max:.1f} mm")
    print(f"  Range:               {depth_range:.1f} mm")
    print(f"  Floor level:         {floor_median:.1f} mm")
    
    # Auto-calculate gap if not provided
    if test_gap_mm is None:
        test_gap_mm = auto_calculate_tomato_gap(depths_detrended, verbose=True)
        print(f"\n💡 Auto-calculated gap: {test_gap_mm:.1f} mm")
    else:
        print(f"\n🔧 Test gap: {test_gap_mm:.1f} mm")
    
    # Test segmentation
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths_detrended[:edge_size], depths_detrended[-edge_size:]]))
    
    tomato_threshold = floor_level - test_gap_mm
    tomato_mask = depths_detrended < tomato_threshold
    
    tomato_count = tomato_mask.sum()
    tomato_percent = 100 * tomato_count / n
    
    print(f"\nSegmentation:")
    print(f"  Floor level:      {floor_level:.1f} mm")
    print(f"  Tomato threshold: {tomato_threshold:.1f} mm")
    print(f"  Tomato points:    {tomato_count} ({tomato_percent:.1f}%)")
    print(f"  Floor points:     {n - tomato_count} ({100 - tomato_percent:.1f}%)")
    
    # Recommendations
    recommended_gap = test_gap_mm
    
    if tomato_percent < 10:
        print(f"\n⚠️ WARNING: Too few tomato points")
        recommended_gap = depth_range * 0.15
        print(f"  💡 Recommendation: Decrease gap to {recommended_gap:.1f} mm")
    elif tomato_percent > 80:
        print(f"\n⚠️ WARNING: Too many tomato points")
        recommended_gap = depth_range * 0.35
        print(f"  💡 Recommendation: Increase gap to {recommended_gap:.1f} mm")
    elif tomato_percent < 20:
        print(f"\n📝 Note: Low tomato coverage")
        recommended_gap = depth_range * 0.18
        print(f"  💡 Consider decreasing gap to {recommended_gap:.1f} mm")
    elif tomato_percent > 60:
        print(f"\n📝 Note: High tomato coverage")
        recommended_gap = depth_range * 0.28
        print(f"  💡 Consider increasing gap to {recommended_gap:.1f} mm")
    else:
        print(f"\n✅ Status: Good segmentation")
    
    print(f"\n{'='*70}")
    print(f"RECOMMENDED VALUES:")
    print(f"{'='*70}")
    print(f"tomato_floor_gap_mm = {recommended_gap:.1f}")
    print(f"auto_calculate_gap  = True  # Or use fixed value above")
    print(f"\n💡 Good range: {depth_range*0.15:.1f} - {depth_range*0.30:.1f} mm")
    print(f"{'='*70}\n")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Detrended profile
    ax1 = axes[0]
    ax1.plot(us_valid, depths_detrended, 'b-', linewidth=1, alpha=0.7, label='Measured')
    ax1.axhline(floor_level, color='red', linewidth=2, linestyle='-', 
                label=f'Floor: {floor_level:.1f} mm')
    ax1.axhline(tomato_threshold, color='green', linewidth=2, linestyle='--',
                label=f'Threshold: {tomato_threshold:.1f} mm')
    ax1.fill_between(us_valid, depth_min, tomato_threshold, alpha=0.2, color='green')
    ax1.set_xlabel('Image column (pixels)')
    ax1.set_ylabel('Depth Z (mm)')
    ax1.set_title(f'Depth Profile (Gap={test_gap_mm:.1f} mm)')
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
    """Example usage of tuning tools."""
    
    # =================================================================
    # CONFIGURATION - Set your test image and parameters
    # =================================================================
    
    IMG_PATH = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\12-11-25\BASE_PICTURE\NATURAL\red_light\2.JPG"
    
    # Cropping settings
    ENABLE_CROP = True
    CROP_X_START = 1700      # Start x pixel
    CROP_X_END = 4300        # End x pixel (-1 = full width)
    CROP_Y_START = 1000      # Start y pixel
    CROP_Y_END = 2900        # End y pixel (-1 = full height)
    
    # Camera settings
    LASER_COLOR = 'blue'
    FX = 4719.1
    FY = 4705.9
    CX = 3000.0
    CY = 2000.0
    
    # Scale correction
    USE_SCALE_CORRECTION = True
    CORRECTION_FACTOR_DEPTH = 0.9578
    CORRECTION_FACTOR_WIDTH = 2.2080
    
    # Laser extraction parameters
    COLOR_RATIO_THRESHOLD = 0.49
    MIN_VAL_FRACTION = 0.24
    BANDPASS_KERNEL = 9
    
    # =================================================================
    
    # Tune laser extraction
    print("\n🔍 STARTING LASER EXTRACTION TUNING...")
    recommended_laser = tune_laser_extraction(
        img_path=IMG_PATH,
        laser_color=LASER_COLOR,
        color_ratio_threshold=COLOR_RATIO_THRESHOLD,
        min_val_fraction=MIN_VAL_FRACTION,
        bandpass_kernel=BANDPASS_KERNEL,
        enable_crop=ENABLE_CROP,
        crop_x_start=CROP_X_START,
        crop_x_end=CROP_X_END,
        crop_y_start=CROP_Y_START,
        crop_y_end=CROP_Y_END,
        fx=FX, fy=FY, cx=CX, cy=CY
    )
    
    # Tune segmentation
    # print("\n🔍 STARTING SEGMENTATION TUNING...")
    # recommended_gap = tune_segmentation(
    #     img_path=IMG_PATH,
    #     laser_color=LASER_COLOR,
    #     color_ratio_threshold=COLOR_RATIO_THRESHOLD,
    #     min_val_fraction=MIN_VAL_FRACTION,
    #     bandpass_kernel=BANDPASS_KERNEL,
    #     correction_factor_depth=CORRECTION_FACTOR_DEPTH,
    #     correction_factor_width=CORRECTION_FACTOR_WIDTH,
    #     use_scale_correction=USE_SCALE_CORRECTION,
    #     test_gap_mm=None,  # Auto-calculate
    #     enable_crop=ENABLE_CROP,
    #     crop_x_start=CROP_X_START,
    #     crop_x_end=CROP_X_END,
    #     crop_y_start=CROP_Y_START,
    #     crop_y_end=CROP_Y_END,
    #     fx=FX, fy=FY, cx=CX, cy=CY
    # )
    
    # print("\n" + "="*70)
    # print("TUNING COMPLETE - SUMMARY")
    # print("="*70)
    # print("Copy these values to your main configuration:")
    # print()
    print(f"# Laser Extraction")
    print(f"LASER_COLOR = '{recommended_laser['laser_color']}'")
    print(f"COLOR_RATIO_THRESHOLD = {recommended_laser['color_ratio_threshold']:.2f}")
    print(f"MIN_VAL_FRACTION = {recommended_laser['min_val_fraction']:.2f}")
    print(f"BANDPASS_KERNEL = {recommended_laser['bandpass_kernel']}")
    print()
    # print(f"# Segmentation")
    # if recommended_gap:
    #     print(f"TOMATO_FLOOR_GAP_MM = {recommended_gap:.1f}")
    # print("="*70 + "\n")


if __name__ == "__main__":
    main()