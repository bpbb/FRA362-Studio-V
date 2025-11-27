"""
Segmentation module for tomato defect detection.
Handles tomato segmentation, reference surface creation, and defect size calculation.
"""

import numpy as np
import csv
from scipy.ndimage import gaussian_filter1d, median_filter, label as ndimage_label
from scipy.optimize import least_squares


# =========================
# EDGE CUTTING (SLOPE-BASED)
# =========================

def find_tomato_edges_by_slope(us, depths, slope_threshold_mm_per_px=0.5, 
                                min_flat_points=20, smoothing_window=5,
                                verbose=True):
    """
    Find the real tomato surface edges by detecting steep slopes.
    
    Scans from left and right edges toward center to find where the
    steep transition (floor-to-tomato) ends and flat surface begins.
    
    Args:
        us: X pixel coordinates
        depths: Depth values (should be valid, no NaN)
        slope_threshold_mm_per_px: Threshold for "steep" slope
        min_flat_points: Minimum consecutive flat points to confirm edge
        smoothing_window: Window for smoothing before slope calculation
        verbose: Print debug info
    
    Returns:
        left_idx: Index where left tomato surface starts
        right_idx: Index where right tomato surface ends
        slopes: Calculated slopes for visualization
    """
    n = len(depths)
    
    if n < min_flat_points * 2:
        if verbose:
            print(f"   Edge cutting: Too few points ({n}), skipping")
        return 0, n - 1, None
    
    # Smooth depths for more stable slope calculation
    if smoothing_window > 1:
        depths_smooth = gaussian_filter1d(depths, sigma=smoothing_window, mode='nearest')
    else:
        depths_smooth = depths.copy()
    
    # Calculate slopes (change in depth per pixel)
    # Positive slope = going deeper (toward floor)
    # Negative slope = going shallower (toward camera/tomato top)
    slopes = np.diff(depths_smooth)
    slopes = np.concatenate([[0], slopes])  # Pad to same length
    
    # Also calculate absolute slope for detecting any steep region
    abs_slopes = np.abs(slopes)
    
    # === Find LEFT edge (scan from left toward center) ===
    left_idx = 0
    flat_count = 0
    
    for i in range(n // 2):  # Only scan first half
        if abs_slopes[i] < slope_threshold_mm_per_px:
            flat_count += 1
            if flat_count >= min_flat_points:
                # Found stable flat region - this is where tomato surface starts
                left_idx = max(0, i - min_flat_points + 1)
                break
        else:
            flat_count = 0  # Reset counter when steep slope found
    
    # === Find RIGHT edge (scan from right toward center) ===
    right_idx = n - 1
    flat_count = 0
    
    for i in range(n - 1, n // 2, -1):  # Only scan second half
        if abs_slopes[i] < slope_threshold_mm_per_px:
            flat_count += 1
            if flat_count >= min_flat_points:
                # Found stable flat region - this is where tomato surface ends
                right_idx = min(n - 1, i + min_flat_points - 1)
                break
        else:
            flat_count = 0  # Reset counter when steep slope found
    
    # Validate results
    if left_idx >= right_idx:
        if verbose:
            print(f"   Edge cutting: Invalid edges found (left={left_idx}, right={right_idx}), using full range")
        return 0, n - 1, slopes
    
    if verbose:
        original_width = n
        new_width = right_idx - left_idx + 1
        cut_left = left_idx
        cut_right = n - 1 - right_idx
        print(f"\nEdge Cutting (Slope-based):")
        print(f"   Slope threshold: {slope_threshold_mm_per_px:.2f} mm/px")
        print(f"   Original points: {original_width}")
        print(f"   Left cut: {cut_left} points (steep edge)")
        print(f"   Right cut: {cut_right} points (steep edge)")
        print(f"   Remaining: {new_width} points ({100*new_width/original_width:.1f}%)")
        print(f"   Index range: [{left_idx} : {right_idx}]")
    
    return left_idx, right_idx, slopes


def cut_tomato_edges(us, depths, tomato_mask, params, verbose=True):
    """
    Cut steep edges from tomato region to get only the real surface.
    
    This function:
    1. Extracts the tomato region
    2. Finds steep slopes at left and right edges
    3. Applies offset to move cuts inward
    4. Updates the tomato_mask to exclude steep edge regions
    
    Args:
        us: X pixel coordinates (full array)
        depths: Depth values (full array)
        tomato_mask: Boolean mask of tomato region
        params: Params object with edge cutting settings
        verbose: Print debug info
    
    Returns:
        tomato_mask_cut: Updated mask with edges removed
        cut_info: Dict with cutting information
    """
    # Get edge cutting parameters from params (with defaults)
    slope_threshold = getattr(params, 'edge_slope_threshold', 0.5)
    min_flat_points = getattr(params, 'edge_min_flat_points', 20)
    smoothing_window = getattr(params, 'edge_smoothing_window', 5)
    enable_edge_cutting = getattr(params, 'enable_edge_cutting', True)
    edge_cut_offset = getattr(params, 'edge_cut_offset', 0)  # New: offset to move cuts inward
    
    if not enable_edge_cutting:
        if verbose:
            print(f"\nEdge cutting: Disabled")
        return tomato_mask, None
    
    # Extract tomato region
    tomato_indices = np.where(tomato_mask)[0]
    
    if len(tomato_indices) < 50:
        if verbose:
            print(f"\nEdge cutting: Too few tomato points ({len(tomato_indices)}), skipping")
        return tomato_mask, None
    
    # Get tomato data
    us_tomato = us[tomato_indices]
    depths_tomato = depths[tomato_indices]
    
    # Find edges by slope
    left_local, right_local, slopes = find_tomato_edges_by_slope(
        us_tomato, depths_tomato,
        slope_threshold_mm_per_px=slope_threshold,
        min_flat_points=min_flat_points,
        smoothing_window=smoothing_window,
        verbose=False  # We'll print our own message
    )
    
    # Apply offset to move cuts inward (toward center)
    left_local_offset = left_local + edge_cut_offset
    right_local_offset = right_local - edge_cut_offset
    
    # Make sure we don't cross the middle
    if left_local_offset >= right_local_offset:
        if verbose:
            print(f"\nEdge cutting: Offset too large, cuts would cross. Using original cuts.")
        left_local_offset = left_local
        right_local_offset = right_local
    
    # Validate
    if left_local_offset >= right_local_offset:
        if verbose:
            print(f"\nEdge cutting: Invalid edges, using full range")
        return tomato_mask, None
    
    if verbose:
        original_width = len(tomato_indices)
        new_width = right_local_offset - left_local_offset + 1
        print(f"\nEdge Cutting:")
        print(f"   Original tomato points: {original_width}")
        print(f"   Slope threshold: {slope_threshold:.2f} mm/px")
        print(f"   Offset: {edge_cut_offset} points inward")
        print(f"   Left cut at local index: {left_local} -> {left_local_offset}")
        print(f"   Right cut at local index: {right_local} -> {right_local_offset}")
        print(f"   Remaining: {new_width} points ({100*new_width/original_width:.1f}%)")
    
    # Create new mask with edges cut
    tomato_mask_cut = tomato_mask.copy()
    
    # Remove left edge points (including offset)
    if left_local_offset > 0:
        left_global_indices = tomato_indices[:left_local_offset]
        tomato_mask_cut[left_global_indices] = False
    
    # Remove right edge points (including offset)
    if right_local_offset < len(tomato_indices) - 1:
        right_global_indices = tomato_indices[right_local_offset + 1:]
        tomato_mask_cut[right_global_indices] = False
    
    # Store cutting info for visualization
    cut_info = {
        'left_local_idx': left_local_offset,
        'right_local_idx': right_local_offset,
        'left_global_idx': tomato_indices[left_local_offset] if left_local_offset < len(tomato_indices) else None,
        'right_global_idx': tomato_indices[right_local_offset] if right_local_offset < len(tomato_indices) else None,
        'left_u': us_tomato[left_local_offset] if left_local_offset < len(us_tomato) else None,
        'right_u': us_tomato[right_local_offset] if right_local_offset < len(us_tomato) else None,
        'slopes': slopes,
        'original_count': len(tomato_indices),
        'new_count': tomato_mask_cut.sum(),
        'offset_applied': edge_cut_offset,
    }
    
    return tomato_mask_cut, cut_info


# =========================
# SEGMENTATION
# =========================

def auto_calculate_tomato_gap(depths, verbose=True):
    """
    Automatically calculate optimal tomato_floor_gap_mm based on depth data.
    
    Args:
        depths: Array of depth values (should be detrended)
        verbose: Print calculation details
    
    Returns:
        float: Recommended gap value in mm
    """
    n = len(depths)
    
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths[:edge_size], depths[-edge_size:]]))
    
    depth_min = depths.min()
    depth_max = depths.max()
    depth_range = depth_max - depth_min
    
    initial_gap = depth_range * 0.20
    
    tomato_threshold = floor_level - initial_gap
    tomato_mask_test = depths < tomato_threshold
    tomato_count = tomato_mask_test.sum()
    tomato_percent = 100 * tomato_count / n
    
    if tomato_percent < 10:
        recommended_gap = depth_range * 0.18
        status = "Too few tomato points - decreased gap"
    elif tomato_percent > 70:
        recommended_gap = depth_range * 0.40
        status = "Too many tomato points - increased gap"
    elif tomato_percent < 25:
        recommended_gap = depth_range * 0.22
        status = "Low tomato coverage - slightly decreased gap"
    elif tomato_percent > 55:
        recommended_gap = depth_range * 0.32
        status = "High tomato coverage - slightly increased gap"
    else:
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
    
    Args:
        depths: Depth values (detrended)
        params: Params object
    
    Returns:
        tomato_mask: Boolean mask
        floor_level: Estimated floor depth
    """
    n = len(depths)
    
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths[:edge_size], depths[-edge_size:]]))
    
    if params.auto_calculate_gap:
        gap_mm = auto_calculate_tomato_gap(depths, verbose=True)
    else:
        gap_mm = params.tomato_floor_gap_mm
        print(f"\nSegmentation:")
        print(f"   Using fixed gap: {gap_mm:.1f} mm")
    
    print(f"   Floor level: {floor_level:.1f} mm")
    print(f"   Depth range: {depths.min():.1f} - {depths.max():.1f} mm")
    print(f"   Using gap threshold: {gap_mm:.1f} mm")
    
    tomato_mask = depths < (floor_level - gap_mm)
    
    labeled, num = ndimage_label(tomato_mask)
    
    if num > 0:
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
# REFERENCE SURFACE FITTING
# =========================

def fit_circular_arc(us, depths):
    """
    Fit a circular arc to points (u, depth).
    
    Circle equation: (u - u_c)^2 + (d - d_c)^2 = R^2
    
    Returns:
        u_arc, d_arc: Arc coordinates for all u values
        (u_c, d_c, R): Circle parameters
    """
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
    """
    Create reference surface using healthy points (where deviation < threshold).
    
    TWO-PASS APPROACH:
    1. First pass: Create rough reference from outer edges
    2. Calculate deviation, SMOOTH it, find healthy points (smoothed deviation < threshold)
    3. Second pass: Fit final reference through SMOOTHED healthy points
    
    Args:
        us: Pixel coordinates
        depths: Depth values
        params: Parameters
        rough_defect_mask: Optional boolean mask (not used)
    
    Returns:
        reference: Fitted reference surface
        edge_mask: Boolean mask of healthy points used for fitting
        degree_name: Name of fit type
    """
    n = len(us)
    
    print(f"\nReference Surface (Two-Pass with Smoothed Deviation):")
    print(f"   Total points: {n}")
    
    # === SMOOTH the depths first ===
    smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
    depths_smooth = gaussian_filter1d(depths, sigma=smoothing_sigma, mode='nearest')
    
    print(f"   Smoothing depths with sigma = {smoothing_sigma}")
    
    # === PASS 1: Rough reference from outer edges (using smoothed depths) ===
    edge_size = max(10, int(n * params.edge_region_percent / 100.0))
    
    rough_edge_us = np.concatenate([us[:edge_size], us[-edge_size:]])
    rough_edge_depths = np.concatenate([depths_smooth[:edge_size], depths_smooth[-edge_size:]])
    
    # Fit rough reference (quadratic polynomial)
    rough_coeffs = np.polyfit(rough_edge_us, rough_edge_depths, 2)
    rough_reference = np.polyval(rough_coeffs, us)
    
    print(f"   Pass 1: Rough reference from edges ({edge_size*2} points)")
    
    # === Calculate deviation from rough reference (using smoothed depths) ===
    deviation_smooth = depths_smooth - rough_reference
    
    # Print deviation range for debugging
    print(f"   Smoothed deviation range: {deviation_smooth.min():.2f} to {deviation_smooth.max():.2f} mm")
    
    # === Find healthy points using SMOOTHED deviation ===
    threshold = params.defect_threshold_mm
    healthy_mask = deviation_smooth < threshold
    
    print(f"   Threshold: {threshold:.2f} mm")
    
    # Clean up: remove isolated points
    min_group = 5
    labeled, num_regions = ndimage_label(healthy_mask)
    for region_id in range(1, num_regions + 1):
        region_size = (labeled == region_id).sum()
        if region_size < min_group:
            healthy_mask[labeled == region_id] = False
    
    healthy_count = healthy_mask.sum()
    healthy_percent = 100 * healthy_count / n
    
    print(f"   Found {healthy_count} healthy points ({healthy_percent:.1f}%) where smoothed deviation < {threshold}mm")
    
    # === PASS 2: Fit final reference through SMOOTHED healthy points ===
    if healthy_count >= 15:
        healthy_us = us[healthy_mask]
        healthy_depths = depths_smooth[healthy_mask]  # Use SMOOTHED depths!
        
        if params.use_circular_reference and len(healthy_us) >= 10:
            try:
                u_arc, d_arc, (u_c, d_c, R) = fit_circular_arc(healthy_us, healthy_depths)
                reference = np.interp(us, u_arc, d_arc, left=np.nan, right=np.nan)
                
                # Fill NaN at edges
                valid_ref = ~np.isnan(reference)
                if not valid_ref.all() and valid_ref.any():
                    first_valid = np.where(valid_ref)[0][0]
                    last_valid = np.where(valid_ref)[0][-1]
                    if first_valid > 0:
                        reference[:first_valid] = reference[first_valid]
                    if last_valid < n - 1:
                        reference[last_valid+1:] = reference[last_valid]
                
                degree_name = "circular arc"
                print(f"   Pass 2: Circular arc fit through smoothed healthy points")
                print(f"   Circle: center=({u_c:.1f}, {d_c:.1f}), R={R:.1f}mm")
            except Exception as e:
                print(f"   Circular fit failed: {e}, using polynomial")
                coeffs = np.polyfit(healthy_us, healthy_depths, params.edge_poly_degree)
                reference = np.polyval(coeffs, us)
                degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
                    params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
                )
        else:
            print(f"   Pass 2: Polynomial (degree {params.edge_poly_degree}) fit")
            coeffs = np.polyfit(healthy_us, healthy_depths, params.edge_poly_degree)
            reference = np.polyval(coeffs, us)
            degree_name = {1: "linear", 2: "quadratic", 3: "cubic", 5: "quintic"}.get(
                params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
            )
        
        edge_mask = healthy_mask
        
    else:
        # Fallback: use rough reference
        print(f"   WARNING: Only {healthy_count} healthy points, using rough reference")
        reference = rough_reference
        edge_mask = np.zeros(n, dtype=bool)
        edge_mask[:edge_size] = True
        edge_mask[-edge_size:] = True
        degree_name = "quadratic"
    
    return reference, edge_mask, degree_name


def _find_defect_boundaries_by_scanning(deviation, threshold, buffer_points=15, smoothing_sigma=15):
    """
    Find defect boundaries by scanning from edges toward center.
    
    IMPROVED: 
    1. Apply strong smoothing to deviation to remove noise
    2. Look for where deviation leaves/returns to baseline
    
    Args:
        deviation: Array of deviation values
        threshold: Deviation threshold for defect detection
        buffer_points: Safety buffer to exclude transition zones
        smoothing_sigma: Gaussian smoothing sigma (higher = smoother)
    
    Returns:
        left_end: Last index of left healthy region (exclusive)
        right_start: First index of right healthy region (inclusive)
        deviation_smooth: Smoothed deviation for visualization
    """
    n = len(deviation)
    
    # Apply STRONG smoothing to remove noise from deviation
    deviation_smooth = gaussian_filter1d(deviation, sigma=smoothing_sigma, mode='nearest')
    
    # Also apply median filter to remove spikes
    median_window = max(5, n // 100)
    if median_window % 2 == 0:
        median_window += 1  # Must be odd
    deviation_smooth = median_filter(deviation_smooth, size=median_window)
    
    # Define baseline threshold - should be close to 0
    baseline_threshold = threshold * 0.4  # 40% of defect threshold
    
    # --- LEFT SCAN: Find where surface leaves baseline ---
    consecutive_threshold = 5
    consecutive_above_baseline = 0
    defect_start_left = n // 2
    
    for i in range(n):
        if deviation_smooth[i] > baseline_threshold:
            consecutive_above_baseline += 1
            if consecutive_above_baseline >= consecutive_threshold:
                defect_start_left = i - consecutive_threshold + 1
                break
        else:
            consecutive_above_baseline = 0
    
    # Apply buffer
    left_end = max(0, defect_start_left - buffer_points)
    
    # --- RIGHT SCAN: Find where surface returns to baseline ---
    consecutive_above_baseline = 0
    defect_end_right = n // 2
    
    for i in range(n - 1, -1, -1):
        if deviation_smooth[i] > baseline_threshold:
            consecutive_above_baseline += 1
            if consecutive_above_baseline >= consecutive_threshold:
                defect_end_right = i + consecutive_threshold - 1
                break
        else:
            consecutive_above_baseline = 0
    
    # Apply buffer
    right_start = min(n, defect_end_right + buffer_points + 1)
    
    # Validate
    if left_end >= right_start:
        left_end = n // 4
        right_start = 3 * n // 4
    
    return left_end, right_start, deviation_smooth


def _find_flat_healthy_regions(us, depths, defect_mask=None, 
                                slope_threshold=0.3, min_region_size=8):
    """
    Find flat healthy regions for reference fitting.
    (Legacy function - kept for compatibility)
    """
    n = len(depths)
    
    if n < min_region_size * 2:
        mask = np.zeros(n, dtype=bool)
        edge_size = max(5, n // 4)
        mask[:edge_size] = True
        mask[-edge_size:] = True
        return mask
    
    depths_smooth = gaussian_filter1d(depths, sigma=3, mode='nearest')
    slopes = np.abs(np.diff(depths_smooth))
    slopes = np.concatenate([[0], slopes])
    flat_mask = slopes < slope_threshold
    
    if defect_mask is not None:
        flat_mask = flat_mask & ~defect_mask
    
    if flat_mask.any():
        baseline_depth = np.percentile(depths[flat_mask], 20)
    else:
        baseline_depth = np.percentile(depths, 15)
    
    near_baseline = np.abs(depths - baseline_depth) < 1.5
    healthy_mask = flat_mask & near_baseline
    
    labeled, num_regions = ndimage_label(healthy_mask)
    for i in range(1, num_regions + 1):
        region_size = (labeled == i).sum()
        if region_size < min_region_size:
            healthy_mask[labeled == i] = False
    
    return healthy_mask


def _create_standard_edge_reference(us, depths, params):
    """Create reference using standard edge-based fitting (no defect info)."""
    n = len(us)
    edge_size = max(5, int(n * params.edge_region_percent / 100.0))
    edge_mask = np.zeros(n, dtype=bool)
    edge_mask[:edge_size] = True
    edge_mask[-edge_size:] = True
    
    print(f"\nReference Surface (Standard Edges):")
    
    edge_us = us[edge_mask]
    edge_depths = depths[edge_mask]
    
    # Filter out boundary spikes
    edge_gradients = np.abs(np.diff(edge_depths))
    if len(edge_gradients) > 0:
        spike_threshold = 5.0
        good_start = 0
        for i in range(min(5, len(edge_gradients))):
            if edge_gradients[i] > spike_threshold:
                good_start = i + 2
            else:
                break
        
        good_end = len(edge_depths)
        for i in range(max(0, len(edge_gradients) - 5), len(edge_gradients)):
            if edge_gradients[i] > spike_threshold:
                good_end = i
                break
        
        if good_start < good_end and (good_end - good_start) > 10:
            edge_us = edge_us[good_start:good_end]
            edge_depths = edge_depths[good_start:good_end]
            print(f"   Filtered boundary spikes: using points {good_start} to {good_end}")
    
    if params.use_circular_reference and len(edge_us) >= 10:
        print(f"   Method: Circular arc fit to tomato edges")
        print(f"   Edge regions: {edge_size*2}/{n} points ({params.edge_region_percent*2:.0f}%)")
        
        try:
            u_arc, d_arc, (u_c, d_c, R) = fit_circular_arc(edge_us, edge_depths)
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
    
    Returns:
        defects: List of defect dictionaries
        defect_mask_full: Boolean mask of defect regions
        reference: Reference surface
        edge_mask_full: Mask of edge regions used
        degree_name: Reference method name
    """
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
            
            depths_clean = median_filter(depths_tomato, size=window, mode='nearest')
            print(f"   Median filter: window={window}")
        else:
            depths_clean = depths_tomato.copy()
        
        depths_processed = gaussian_filter1d(depths_clean, sigma=params.depth_smoothing_sigma, mode='nearest')
        print(f"   Gaussian smoothing: sigma={params.depth_smoothing_sigma:.1f}")
    else:
        depths_processed = depths_tomato.copy()
        print(f"\nNo pre-smoothing applied")
    
    # PASS 1: Rough defect detection
    print(f"\n{'='*60}")
    print(f"PASS 1: Initial defect detection")
    print(f"{'='*60}")
    
    ref_rough, edge_mask_rough, _ = create_defect_adjacent_reference(
        us_tomato, depths_processed, params, rough_defect_mask=None
    )
    
    dev_rough = depths_processed - ref_rough
    
    # Apply full smoothing to match visualization
    smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
    dev_rough = gaussian_filter1d(dev_rough, sigma=smoothing_sigma, mode='nearest')
    
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
        # PASS 2: Refined detection
        print(f"\n{'='*60}")
        print(f"PASS 2: Refined defect detection")
        print(f"{'='*60}")
        
        ref_tomato, edge_mask_tomato, degree_name = create_defect_adjacent_reference(
            us_tomato, depths_processed, params, rough_defect_mask=rough_defect_mask
        )
        
        dev_tomato = depths_processed - ref_tomato
        
        # Apply full smoothing to match visualization (dark blue line)
        smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
        dev_tomato_smooth = gaussian_filter1d(dev_tomato, sigma=smoothing_sigma, mode='nearest')
        print(f"   Deviation smoothing: sigma={smoothing_sigma}")
        
        # Use smoothed deviation for defect detection
        defect_mask_tomato = dev_tomato_smooth > params.defect_threshold_mm
        
        if params.edge_exclude_percent > 0:
            edge_margin = int(len(depths_tomato) * params.edge_exclude_percent / 100.0)
            if edge_margin > 0:
                defect_mask_tomato[:edge_margin] = False
                defect_mask_tomato[-edge_margin:] = False
        
        # Store smoothed deviation for defect measurements
        dev_tomato = dev_tomato_smooth
    
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
                'width_mm': len(indices) * params.pixel_size_mm,
                'max_depth_mm': float(defect_depths.max()),
                'mean_depth_mm': float(defect_depths.mean()),
                'indices': indices
            })
    
    # Sort by maximum depth
    defects.sort(key=lambda d: -d['max_depth_mm'])
    
    for i, d in enumerate(defects, 1):
        d['id'] = i
    
    print(f"   Detected: {len(defects)} defects after filtering")
    
    return defects, defect_mask_full, reference, edge_mask_full, degree_name


# =========================
# CALIBRATION
# =========================

def apply_calibration_equations(defects, params):
    """
    Apply polynomial calibration equations to defect measurements.
    
    Args:
        defects: List of defect dictionaries
        params: Params with calibration coefficients
    
    Returns:
        Calibrated defects list
    """
    if not params.use_calibration_equations:
        return defects
    
    if not params.calibration_width_coeffs and not params.calibration_depth_coeffs:
        return defects
    
    calibrated_defects = []
    
    for defect in defects:
        calib_defect = defect.copy()
        
        if params.calibration_width_coeffs:
            measured_width = defect['width_mm']
            calibrated_width = np.polyval(params.calibration_width_coeffs, measured_width)
            calib_defect['width_mm_raw'] = measured_width
            calib_defect['width_mm'] = calibrated_width
        
        if params.calibration_depth_coeffs:
            measured_depth = defect['max_depth_mm']
            measured_mean_depth = defect['mean_depth_mm']
            
            calibrated_max_depth = np.polyval(params.calibration_depth_coeffs, measured_depth)
            calibrated_mean_depth = np.polyval(params.calibration_depth_coeffs, measured_mean_depth)
            
            calib_defect['max_depth_mm_raw'] = measured_depth
            calib_defect['mean_depth_mm_raw'] = measured_mean_depth
            calib_defect['max_depth_mm'] = calibrated_max_depth
            calib_defect['mean_depth_mm'] = calibrated_mean_depth
        
        calibrated_defects.append(calib_defect)
    
    return calibrated_defects


# =========================
# RESULTS EXPORT
# =========================

def save_results(us, vs, depths, tomato_mask, reference, defects, defect_mask, params):
    """Save detailed results to CSV."""
    print(f"\nSaving results...")
    
    defect_ids = np.zeros(len(us), dtype=int)
    tomato_indices = np.where(tomato_mask)[0]
    
    for defect in defects:
        for local_idx in defect['indices']:
            global_idx = tomato_indices[local_idx]
            if defect_mask[global_idx]:
                defect_ids[global_idx] = defect['id']
    
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
    
    summary_file = params.out_csv.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("DEFECT DETECTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        if params.use_calibration_equations:
            f.write("NOTE: Measurements below are CALIBRATED actual values\n")
            f.write("      (Raw measured values shown in parentheses)\n\n")
        
        f.write(f"Total defects: {len(defects)}\n\n")
        
        if defects:
            f.write(f"{'ID':<4} {'Center(px)':<12} {'Width(mm)':<18} "
                   f"{'Max Depth(mm)':<18} {'Mean Depth(mm)':<18}\n")
            f.write("-"*80 + "\n")
            
            for d in defects:
                f.write(f"{d['id']:<4} {d['center_px']:>10.1f}   "
                       f"{d['width_mm']:>8.1f}   "
                       f"{d['max_depth_mm']:>12.2f}   "
                       f"{d['mean_depth_mm']:>12.2f}\n")
    
    print(f"Saved: {params.out_csv}")
    print(f"Saved: {summary_file}")