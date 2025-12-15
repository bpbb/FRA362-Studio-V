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
        # if verbose:
            # print(f"   Edge cutting: Too few points ({n}), skipping")
        return 0, n - 1, None
    
    # Smooth depths for more stable slope calculation
    if smoothing_window > 1:
        depths_smooth = gaussian_filter1d(depths, sigma=smoothing_window, mode='nearest')
    else:
        depths_smooth = depths.copy()
    
    # Calculate slopes (change in depth per pixel)
    slopes = np.diff(depths_smooth)
    slopes = np.concatenate([[0], slopes])
    
    abs_slopes = np.abs(slopes)
    
    # === Find LEFT edge (scan from left toward center) ===
    left_idx = 0
    flat_count = 0
    
    for i in range(n // 2):
        if abs_slopes[i] < slope_threshold_mm_per_px:
            flat_count += 1
            if flat_count >= min_flat_points:
                left_idx = max(0, i - min_flat_points + 1)
                break
        else:
            flat_count = 0
    
    # === Find RIGHT edge (scan from right toward center) ===
    right_idx = n - 1
    flat_count = 0
    
    for i in range(n - 1, n // 2, -1):
        if abs_slopes[i] < slope_threshold_mm_per_px:
            flat_count += 1
            if flat_count >= min_flat_points:
                right_idx = min(n - 1, i + min_flat_points - 1)
                break
        else:
            flat_count = 0
    
    if left_idx >= right_idx:
        # if verbose:
        #     print(f"   Edge cutting: Invalid edges found, using full range")
        return 0, n - 1, slopes
    
    if verbose:
        original_width = n
        new_width = right_idx - left_idx + 1
        cut_left = left_idx
        cut_right = n - 1 - right_idx
        # print(f"\nEdge Cutting (Slope-based):")
        # print(f"   Slope threshold: {slope_threshold_mm_per_px:.2f} mm/px")
        # print(f"   Original points: {original_width}")
        # print(f"   Left cut: {cut_left} points")
        # print(f"   Right cut: {cut_right} points")
        # print(f"   Remaining: {new_width} points ({100*new_width/original_width:.1f}%)")
    
    return left_idx, right_idx, slopes


def cut_tomato_edges(us, depths, tomato_mask, params, verbose=True):
    """
    Cut steep edges from tomato region to get only the real surface.
    
    Args:
        us: X pixel coordinates (full array)
        depths: Depth values (full array)
        tomato_mask: Boolean mask of tomato region
        params: Params object with edge cutting settings
        verbose: Print debug info
    
    Returns:
        tomato_mask_cut: Updated mask with edges removed
        cut_info: Dict with cutting information including tomato_width_mm
    """
    slope_threshold = getattr(params, 'edge_slope_threshold', 0.5)
    min_flat_points = getattr(params, 'edge_min_flat_points', 20)
    smoothing_window = getattr(params, 'edge_smoothing_window', 5)
    enable_edge_cutting = getattr(params, 'enable_edge_cutting', True)
    edge_cut_offset = getattr(params, 'edge_cut_offset', 0)
    
    if not enable_edge_cutting:
        # if verbose:
        #     print(f"\nEdge cutting: Disabled")
        return tomato_mask, None
    
    tomato_indices = np.where(tomato_mask)[0]
    
    if len(tomato_indices) < 50:
        # if verbose:
        #     print(f"\nEdge cutting: Too few tomato points ({len(tomato_indices)}), skipping")
        return tomato_mask, None
    
    us_tomato = us[tomato_indices]
    depths_tomato = depths[tomato_indices]
    
    left_local, right_local, slopes = find_tomato_edges_by_slope(
        us_tomato, depths_tomato,
        slope_threshold_mm_per_px=slope_threshold,
        min_flat_points=min_flat_points,
        smoothing_window=smoothing_window,
        verbose=False
    )
    
    left_local_offset = left_local + edge_cut_offset
    right_local_offset = right_local - edge_cut_offset
    
    if left_local_offset >= right_local_offset:
        # if verbose:
        #     print(f"\nEdge cutting: Offset too large, using original cuts.")
        left_local_offset = left_local
        right_local_offset = right_local
    
    if left_local_offset >= right_local_offset:
        # if verbose:
        #     print(f"\nEdge cutting: Invalid edges, using full range")
        return tomato_mask, None
    
    # Calculate tomato width
    left_u = us_tomato[left_local] if left_local < len(us_tomato) else None
    right_u = us_tomato[right_local] if right_local < len(us_tomato) else None
    
    # Use fixed width if enabled, otherwise calculate
    if params.use_fixed_tomato_width:
        tomato_width_mm = params.fixed_tomato_width_mm
        tomato_width_px = tomato_width_mm / params.pixel_size_mm
        # if verbose:
            # print(f"   Using FIXED tomato width: {tomato_width_mm:.2f} mm")
    else:
        if left_u is not None and right_u is not None:
            tomato_width_px = (right_u - left_u) * 1.4
            tomato_width_mm = tomato_width_px * params.pixel_size_mm
        else:
            tomato_width_px = None
            tomato_width_mm = None
    
    if verbose:
        original_width = len(tomato_indices)
        new_width = right_local_offset - left_local_offset + 1
        # print(f"\nEdge Cutting:")
        # print(f"   Original tomato points: {original_width}")
        # print(f"   Slope threshold: {slope_threshold:.2f} mm/px")
        # print(f"   Offset: {edge_cut_offset} points inward")
        # print(f"   Remaining: {new_width} points ({100*new_width/original_width:.1f}%)")
        # if tomato_width_mm is not None and not params.use_fixed_tomato_width:
        #     print(f"   Calculated tomato width: {tomato_width_px:.1f} px ({tomato_width_mm:.2f} mm)")
    
    tomato_mask_cut = tomato_mask.copy()
    
    if left_local_offset > 0:
        left_global_indices = tomato_indices[:left_local_offset]
        tomato_mask_cut[left_global_indices] = False
    
    if right_local_offset < len(tomato_indices) - 1:
        right_global_indices = tomato_indices[right_local_offset + 1:]
        tomato_mask_cut[right_global_indices] = False
    
    cut_info = {
        'left_local_idx': left_local_offset,
        'right_local_idx': right_local_offset,
        'left_global_idx': tomato_indices[left_local_offset] if left_local_offset < len(tomato_indices) else None,
        'right_global_idx': tomato_indices[right_local_offset] if right_local_offset < len(tomato_indices) else None,
        'left_u': left_u,
        'right_u': right_u,
        'slopes': slopes,
        'original_count': len(tomato_indices),
        'new_count': tomato_mask_cut.sum(),
        'offset_applied': edge_cut_offset,
        'tomato_width_px': tomato_width_px,
        'tomato_width_mm': tomato_width_mm,
    }
    
    return tomato_mask_cut, cut_info


# =========================
# SEGMENTATION
# =========================

def auto_calculate_tomato_gap(depths, verbose=True):
    """Automatically calculate optimal tomato_floor_gap_mm."""
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
    
    # if verbose:
    #     print(f"\nAuto-calculating tomato_floor_gap_mm:")
    #     print(f"   Depth range: {depth_range:.1f} mm")
    #     print(f"   Recommended gap: {recommended_gap:.1f} mm")
    
    return recommended_gap


def segment_tomato(depths, params):
    """Segment tomato from floor using simple threshold."""
    n = len(depths)
    
    edge_size = max(5, int(n * 0.15))
    floor_level = np.median(np.concatenate([depths[:edge_size], depths[-edge_size:]]))
    
    if params.auto_calculate_gap:
        gap_mm = auto_calculate_tomato_gap(depths, verbose=True)
    else:
        gap_mm = params.tomato_floor_gap_mm
    #     print(f"\nSegmentation:")
    #     print(f"   Using fixed gap: {gap_mm:.1f} mm")
    
    # print(f"   Floor level: {floor_level:.1f} mm")
    
    tomato_mask = depths < (floor_level - gap_mm)
    
    labeled, num = ndimage_label(tomato_mask)
    
    if num > 0:
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        largest = np.argmax(sizes) + 1
        tomato_mask = (labeled == largest)
        
        if tomato_mask.sum() < params.min_tomato_points:
            # print(f"   WARNING: Too few points ({tomato_mask.sum()})")
            tomato_mask = np.zeros(n, dtype=bool)
        else:
            tomato_percent = 100 * tomato_mask.sum() / n
            # print(f"   Tomato: {tomato_mask.sum()} points ({tomato_percent:.1f}%)")
    else:
        tomato_mask = np.zeros(n, dtype=bool)
        # print(f"   WARNING: No tomato region found")
    
    return tomato_mask, floor_level


# =========================
# REFERENCE SURFACE FITTING
# =========================

def fit_circular_arc(us, depths):
    """Fit a circular arc to points."""
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
    """Create reference surface using healthy points."""
    n = len(us)
    
    # print(f"\nReference Surface (Two-Pass):")
    # print(f"   Total points: {n}")
    
    smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
    depths_smooth = gaussian_filter1d(depths, sigma=smoothing_sigma, mode='nearest')
    
    edge_size = max(10, int(n * params.edge_region_percent / 100.0))
    
    rough_edge_us = np.concatenate([us[:edge_size], us[-edge_size:]])
    rough_edge_depths = np.concatenate([depths_smooth[:edge_size], depths_smooth[-edge_size:]])
    
    rough_coeffs = np.polyfit(rough_edge_us, rough_edge_depths, 2)
    rough_reference = np.polyval(rough_coeffs, us)
    
    deviation_smooth = depths_smooth - rough_reference
    
    threshold = params.defect_threshold_mm
    healthy_mask = deviation_smooth < threshold
    
    min_group = 5
    labeled, num_regions = ndimage_label(healthy_mask)
    for region_id in range(1, num_regions + 1):
        region_size = (labeled == region_id).sum()
        if region_size < min_group:
            healthy_mask[labeled == region_id] = False
    
    healthy_count = healthy_mask.sum()
    
    if healthy_count >= 15:
        healthy_us = us[healthy_mask]
        healthy_depths = depths_smooth[healthy_mask]
        
        if params.use_circular_reference and len(healthy_us) >= 10:
            try:
                u_arc, d_arc, (u_c, d_c, R) = fit_circular_arc(healthy_us, healthy_depths)
                reference = np.interp(us, u_arc, d_arc, left=np.nan, right=np.nan)
                
                valid_ref = ~np.isnan(reference)
                if not valid_ref.all() and valid_ref.any():
                    first_valid = np.where(valid_ref)[0][0]
                    last_valid = np.where(valid_ref)[0][-1]
                    if first_valid > 0:
                        reference[:first_valid] = reference[first_valid]
                    if last_valid < n - 1:
                        reference[last_valid+1:] = reference[last_valid]
                
                degree_name = "circular arc"
            except:
                coeffs = np.polyfit(healthy_us, healthy_depths, params.edge_poly_degree)
                reference = np.polyval(coeffs, us)
                degree_name = {1: "linear", 2: "quadratic", 3: "cubic"}.get(
                    params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
                )
        else:
            coeffs = np.polyfit(healthy_us, healthy_depths, params.edge_poly_degree)
            reference = np.polyval(coeffs, us)
            degree_name = {1: "linear", 2: "quadratic", 3: "cubic"}.get(
                params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
            )

        final_deviation = depths_smooth - reference
        healthy_mask_refined = final_deviation < threshold
        edge_mask = healthy_mask_refined  # Use refined mask instead
        
    else:
        reference = rough_reference
        edge_mask = np.zeros(n, dtype=bool)
        edge_mask[:edge_size] = True
        edge_mask[-edge_size:] = True
        degree_name = "quadratic"
    
    return reference, edge_mask, degree_name


# =========================
# DEFECT DETECTION
# =========================

def detect_defects(us, depths, tomato_mask, params):
    """Detect defects using two-pass defect-adjacent reference."""
    us_tomato = us[tomato_mask]
    depths_tomato = depths[tomato_mask]
    
    if len(us_tomato) < 10:
        return [], np.zeros(len(us), dtype=bool), np.full(len(us), np.nan), None, None
    
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
    
    ref_rough, edge_mask_rough, _ = create_defect_adjacent_reference(
        us_tomato, depths_processed, params, rough_defect_mask=None
    )
    
    dev_rough = depths_processed - ref_rough
    
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
    
    if num_rough_defects == 0:
        ref_tomato = ref_rough
        edge_mask_tomato = edge_mask_rough
        dev_tomato = dev_rough
        defect_mask_tomato = rough_defect_mask
        degree_name = {1: "linear", 2: "quadratic", 3: "cubic"}.get(
            params.edge_poly_degree, f"degree-{params.edge_poly_degree}"
        )
    else:
        ref_tomato, edge_mask_tomato, degree_name = create_defect_adjacent_reference(
            us_tomato, depths_processed, params, rough_defect_mask=rough_defect_mask
        )
        
        dev_tomato = depths_processed - ref_tomato
        dev_tomato_smooth = gaussian_filter1d(dev_tomato, sigma=smoothing_sigma, mode='nearest')
        
        defect_mask_tomato = dev_tomato_smooth > params.defect_threshold_mm
        
        if params.edge_exclude_percent > 0:
            edge_margin = int(len(depths_tomato) * params.edge_exclude_percent / 100.0)
            if edge_margin > 0:
                defect_mask_tomato[:edge_margin] = False
                defect_mask_tomato[-edge_margin:] = False
        
        dev_tomato = dev_tomato_smooth
    
    reference = np.full(len(us), np.nan)
    reference[tomato_mask] = ref_tomato
    
    defect_mask_full = np.zeros(len(us), dtype=bool)
    defect_mask_full[tomato_mask] = defect_mask_tomato
    
    edge_mask_full = np.zeros(len(us), dtype=bool)
    edge_mask_full[tomato_mask] = edge_mask_tomato
    
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

            # Calculate reference point percentage
    reference_count = edge_mask_full[tomato_mask].sum()
    tomato_count = tomato_mask.sum()
    reference_percentage = (reference_count / tomato_count * 100) if tomato_count > 0 else 0.0
    
    # print(f"\n📊 REFERENCE QUALITY CHECK:")
    # print(f"   Reference points: {reference_count}/{tomato_count} ({reference_percentage:.1f}%)")
    
    is_inverted_curvature = False
    if len(us_tomato) > 20:
        try:
            # Fit quadratic to reference surface to check curvature
            ref_tomato_valid = ~np.isnan(ref_tomato)
            if ref_tomato_valid.sum() > 10:
                coeffs = np.polyfit(us_tomato[ref_tomato_valid], ref_tomato[ref_tomato_valid], 2)
                # coeffs[0] is the coefficient of x^2 (curvature)
                # For normal tomato: negative (∩ shape - high in middle, low at edges)
                # For inverted/wrong: positive (U shape - low in middle, high at edges)
                
                # print(f"coeffs[0]: {coeffs[0]}")
                if coeffs[0] < 1e-5:  # Positive curvature = upright parabola (U-shaped)
                    is_inverted_curvature = True
                    # print(f"   ⚠️ WARNING: Inverted curvature detected (U-shaped reference)")
                    # print(f"   ⚠️ Surface shape anomaly - consider CULL grade")
        except:
            pass  # If fitting fails, skip this check

    # if reference_percentage < params.min_reference_percentage:
    #     print(f"   ⚠️ WARNING: Reference quality LOW (< {params.min_reference_percentage}%)")
    #     print(f"   ⚠️ Results may be UNRELIABLE - consider CULL grade")
    # elif is_inverted_curvature:
    #     print(f"   ⚠️ WARNING: Surface shape anomaly detected")
    #     print(f"   ⚠️ Results may be UNRELIABLE - consider CULL grade")
    # else:
    #     print(f"   ✓ Reference quality acceptable")

    return defects, defect_mask_full, reference, edge_mask_full, degree_name, reference_percentage, is_inverted_curvature


# =========================
# CALIBRATION
# =========================

def apply_calibration_equations(defects, params):
    """Apply polynomial calibration equations to defect measurements."""
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
    # print(f"\nSaving results...")
    
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
    
    # print(f"Saved: {params.out_csv}")