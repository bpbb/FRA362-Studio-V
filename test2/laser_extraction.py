"""
Laser extraction module for tomato defect detection.
Handles laser stripe detection, filtering, and outlier removal.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def extract_laser_color_ratio(img_bgr, params):
    """
    Extract laser using color ratio - supports blue, green, red lasers.
    
    Args:
        img_bgr: BGR image
        params: Params object with laser settings
    
    Returns:
        us, vs: Arrays of subpixel laser positions
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


def initial_lowpass_filter(us, vs, method="savgol", window=11):
    """
    Apply initial lowpass filter to raw stripe data (STAGE 1).
    
    Methods:
        "savgol": Savitzky-Golay filter (preserves shape while smoothing)
        "gaussian": Gaussian filter (simple and effective)
        "moving_avg": Moving average (fastest, most smoothing)
    
    Returns:
        us, vs_filtered: Original us and filtered vs
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
    
    noise_removed = np.std(vs_original - vs_filtered)
    max_change = np.max(np.abs(vs_original - vs_filtered))
    print(f"   Noise reduced: {noise_removed:.2f} px RMS")
    print(f"   Max change: {max_change:.2f} px")
    
    return us, vs_filtered


def remove_outliers_from_stripe(us, vs, method="improved", params=None):
    """
    Remove outliers from extracted laser stripe (STAGE 2).
    
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
        # STAGE 2.1: Remove large jumps
        dv = np.diff(vs)
        dv_pad = np.concatenate([[0], dv])
        
        mad = np.median(np.abs(dv - np.median(dv)))
        threshold_jump = 2.0 * (mad * 1.4826)
        
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
        
        # STAGE 2.2: Local median absolute deviation
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
        
        # STAGE 2.3: Isolated point removal
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
    
    elif method == "median":
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
    Apply final smoothing to stripe positions with Gaussian filter (STAGE 3).
    
    Should use lower sigma since we already have initial lowpass.
    """
    if sigma <= 0 or len(us) < 5:
        print(f"\nFinal smoothing: Skipped (sigma={sigma})")
        return us, vs
    
    print(f"\n{'='*60}")
    print(f"STAGE 3: Final Stripe Smoothing")
    print(f"{'='*60}")
    print(f"   Gaussian filter: sigma={sigma:.1f}")
    
    vs_original = vs.copy()
    vs_smooth = gaussian_filter1d(vs, sigma=sigma, mode='nearest')
    
    smoothing_change = np.std(vs_original - vs_smooth)
    max_change = np.max(np.abs(vs_original - vs_smooth))
    print(f"   RMS change: {smoothing_change:.2f} px")
    print(f"   Max change: {max_change:.2f} px")
    
    return us, vs_smooth


def preprocess_stripe(us, vs, params):
    """
    Apply full preprocessing pipeline to laser stripe.
    
    Pipeline:
        1. Initial lowpass filter (removes high-frequency noise)
        2. Outlier removal (removes anomalous points)
        3. Final smoothing (gentle polish)
    
    Args:
        us, vs: Raw stripe coordinates
        params: Params object
    
    Returns:
        us_clean, vs_clean: Preprocessed stripe coordinates
    """
    # Stage 1: Initial lowpass filter
    if params.enable_initial_lowpass:
        us, vs = initial_lowpass_filter(us, vs, 
                                        method=params.lowpass_method,
                                        window=params.lowpass_window)
    
    # Stage 2: Outlier removal
    if params.enable_outlier_removal:
        us, vs = remove_outliers_from_stripe(us, vs, method=params.outlier_method)
    
    # Stage 3: Final smoothing
    if params.stripe_smoothing_sigma > 0:
        us, vs = smooth_stripe_positions(us, vs, sigma=params.stripe_smoothing_sigma)
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Final clean points: {len(us)}")
    
    return us, vs