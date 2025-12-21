"""
Main module for tomato defect detection.
Orchestrates the full pipeline and provides visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import time

# Import modules
from preprocess import Params, pixels_to_mm, load_and_preprocess_image
from laser_extraction import extract_laser_color_ratio, preprocess_stripe
from triangulate import process_triangulation
from segmentation import (
    segment_tomato, detect_defects, apply_calibration_equations, save_results,
    cut_tomato_edges
)


# =========================
# VISUALIZATION
# =========================

def visualize(img, us, vs, depths, tomato_mask, defects, defect_mask, 
              reference, edge_mask, degree_name, params, cut_info=None, 
              reference_percentage=None, is_inverted_curvature=False):
    """Create comprehensive visualization."""
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    vs_display = vs.copy()
    if params.flip_image_vertical:
        vs_display = img.shape[0] - 1 - vs_display
    
    valid = ~np.isnan(depths)
    
    # ============ Plot 1: Image with defects ============
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    floor_edge_points = getattr(params, 'floor_edge_points', 200)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= floor_edge_points * 2:
        left_floor_indices = valid_indices[:floor_edge_points]
        right_floor_indices = valid_indices[-floor_edge_points:]
        floor_indices = np.concatenate([left_floor_indices, right_floor_indices])
        ax1.plot(us[floor_indices], vs_display[floor_indices], 'gray', linewidth=2, alpha=0.6, label='Floor')
    
    ax1.plot(us[tomato_mask], vs_display[tomato_mask], 'lime', linewidth=3, alpha=0.8, label='Tomato')
    
    used_positions = []

    # Get tomato width for percentage calculation
    tomato_width_mm = cut_info.get('tomato_width_mm') if cut_info else None

    # Check if grade is CULL (low reference quality OR inverted curvature)
    is_cull = ((reference_percentage is not None and 
                reference_percentage < params.min_reference_percentage) or 
               is_inverted_curvature)
    
    for d in defects:
        mask = defect_mask & (us >= d['start_px']) & (us <= d['end_px'])
        if not mask.any():
            continue
        
        us_def = us[mask]
        vs_def = vs_display[mask]
        v_min = vs_def.min()
        
        # CULL grade overrides all color criteria
        if is_cull:
            color = '#808080'  # Gray - CULL (unreliable detection)
        elif tomato_width_mm and tomato_width_mm > 0:
            # Color based on percentage criteria
            width_percent = (d['width_mm'] / tomato_width_mm) * 100
            depth_percent = (d['mean_depth_mm'] / tomato_width_mm) * 100
            
            if width_percent <= 29.69 and depth_percent <= 4.69:
                color = '#90EE90'  # Light green - FRESH
            elif width_percent <= 39.06 and depth_percent <= 9.38:
                color = '#FFD700'  # Yellow - PROCESSING
            else:
                color = '#FF0000'  # Red - UNUSABLE
        else:
            color = '#FFD700'  # Default yellow
        
        defect_center_u = d['center_px']
        label_v = v_min - 300
        min_spacing = 300
        
        for prev_u, prev_v in used_positions:
            if abs(defect_center_u - prev_u) < 200 and abs(label_v - prev_v) < min_spacing:
                if label_v > img.shape[0] / 2:
                    label_v = prev_v - min_spacing
                else:
                    label_v = prev_v + min_spacing
        
        label_v = max(50, min(label_v, img.shape[0] - 300))
        used_positions.append((defect_center_u, label_v))
        
        ax1.plot([defect_center_u, defect_center_u], 
                [label_v + 20, v_min - 5],
                color=color, linewidth=2, alpha=0.7, zorder=4)
        
        ax1.text(defect_center_u, label_v, 
                f"{d['id']}) D = {d['mean_depth_mm']:.1f} mm, W = {d['width_mm']:.1f} mm", 
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=color, 
                         edgecolor='white', linewidth=3, alpha=0.95),
                zorder=4)
    
    # Add tomato info text box at bottom left
    if tomato_width_mm and tomato_width_mm > 0 and defects:
        # Find max percentages in this image's defects
        max_width_percent = max((d['width_mm'] / tomato_width_mm) * 100 for d in defects)
        max_depth_percent = max((d['mean_depth_mm'] / tomato_width_mm) * 100 for d in defects)
        
        info_text = f"Tomato Width: {tomato_width_mm:.1f} mm\n"
        info_text += f"Max Width: {max_width_percent:.2f}%\n"
        info_text += f"Max Depth: {max_depth_percent:.2f}%"
        
        ax1.text(0.03, 0.03, info_text,
                transform=ax1.transAxes,
                fontsize=8,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='white', 
                        edgecolor='black',
                        linewidth=1,
                        alpha=0.8),
                zorder=10)
    
    ax1.set_title(f"Detected Defects: {len(defects)}", fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.axis('off')
    
    # ============ Plot 2: Depth Profile ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    us_valid = us[valid]
    depths_valid = depths[valid]
    tomato_valid = tomato_mask[valid]
    edge_valid = edge_mask[valid] if edge_mask is not None else np.zeros(valid.sum(), dtype=bool)
    
    smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
    depths_valid_smooth = gaussian_filter1d(depths_valid, sigma=smoothing_sigma, mode='nearest')
    
    us_mm = pixels_to_mm(us_valid, params.cx, params.pixel_size_mm)
    
    ax2.plot(us_mm[tomato_valid], depths_valid[tomato_valid], 
            'blue', linewidth=2, label='Measured (tomato)', zorder=2)
    
    if len(valid_indices) >= floor_edge_points * 2:
        us_floor_mm = pixels_to_mm(us[floor_indices], params.cx, params.pixel_size_mm)
        ax2.plot(us_floor_mm, depths[floor_indices],
                'gray', linewidth=1, alpha=0.5, label='Floor', zorder=1)
    
    if edge_valid.any():
        ax2.scatter(us_mm[edge_valid & tomato_valid], depths_valid_smooth[edge_valid & tomato_valid],
                   c='green', s=10, alpha=0.7, marker='s', label='Reference points', zorder=3)
    
    ref_valid = reference[valid]
    ref_mask = ~np.isnan(ref_valid)
    ax2.plot(us_mm[ref_mask], ref_valid[ref_mask],
            'orange', linewidth=2, linestyle='--', label='Reference', zorder=4)
    
    for d in defects:
        mask = defect_mask[valid] & (us_valid >= d['start_px']) & (us_valid <= d['end_px'])
        if mask.any():
            color = '#FF0000' if d['max_depth_mm'] > 10.0 else '#FFA500' if d['max_depth_mm'] > 5.0 else '#FFD700'
            ax2.scatter(us_mm[mask], depths_valid_smooth[mask], c=color, s=20, 
                       marker='v', edgecolors='darkred', linewidths=0.5, zorder=5)
    
    if cut_info is not None and cut_info.get('left_u') is not None:
        left_u_mm = pixels_to_mm(cut_info['left_u'], params.cx, params.pixel_size_mm)
        right_u_mm = pixels_to_mm(cut_info['right_u'], params.cx, params.pixel_size_mm)
        ax2.axvline(x=left_u_mm, color='red', linewidth=2, linestyle='--', 
                   alpha=0.4, label='Edge cuts')
        ax2.axvline(x=right_u_mm, color='red', linewidth=2, linestyle='--', alpha=0.4)
    
    ax2.set_xlabel('Position (mm)', fontsize=11)
    ax2.set_ylabel('Depth Z (mm)', fontsize=11)
    ax2.set_title(f'{degree_name.title()} reference fit', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    if tomato_valid.any():
        tomato_depths = depths_valid[tomato_valid]
        depth_min = tomato_depths.min()
        depth_max = tomato_depths.max()
        depth_range = depth_max - depth_min
        
        margin_top = depth_range * 0.60
        margin_bottom = depth_range * 0.15
        y_min = depth_min - margin_top
        y_max = depth_max + margin_bottom
        ax2.set_ylim(y_max, y_min)
        
        tomato_us_mm = us_mm[tomato_valid]
        x_min = tomato_us_mm.min()
        x_max = tomato_us_mm.max()
        x_range = x_max - x_min
        x_margin = x_range * 0.1
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
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
            ax3.plot(us_tomato[valid_dev], dev_tomato[valid_dev], 
                    'blue', linewidth=1.5, alpha=0.5, label='Deviation (raw)')
            
            dev_smooth = gaussian_filter1d(dev_tomato[valid_dev], sigma=smoothing_sigma, mode='nearest')
            ax3.plot(us_tomato[valid_dev], dev_smooth, 
                    'darkblue', linewidth=2, label='Deviation (smoothed)')
            
            ax3.axhline(0, color='green', linewidth=2, alpha=0.6, label='Baseline')
            ax3.axhline(params.defect_threshold_mm, color='red', linewidth=2, 
                       linestyle='--', label=f'Threshold ({params.defect_threshold_mm}mm)')
            
            if edge_mask is not None:
                edge_tomato = edge_mask[tomato_mask] & valid_dev
                edge_indices_in_valid = np.where(edge_tomato[valid_dev])[0]
                ax3.scatter(us_tomato[valid_dev][edge_indices_in_valid], 
                          dev_smooth[edge_indices_in_valid],
                          c='green', s=40, alpha=0.5, marker='s', zorder=5,
                          label='Reference points')
            
            for d in defects:
                mask = (us_tomato >= d['start_px']) & (us_tomato <= d['end_px']) & valid_dev
                if mask.any():
                    color = '#FF0000' if d['max_depth_mm'] > 2.0 else '#FFA500' if d['max_depth_mm'] > 1.0 else '#FFD700'
                    ax3.fill_between(us_tomato[mask], 0, dev_tomato[mask], color=color, alpha=0.3)
    
    ax3.set_xlabel('Image column (pixels)', fontsize=11)
    ax3.set_ylabel('Deviation (mm)', fontsize=11)
    ax3.set_title('Deviation Analysis', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_results(defects, params, tomato_width_mm=None, reference_percentage=None, is_inverted_curvature=False):
    """Print detection results to console with percentage-based grading."""
    print(f"\n{'='*70}")
    print(f"  RESULTS: {len(defects)} DEFECTS DETECTED")
    print(f"{'='*70}")
    
    # Check reference quality
    if reference_percentage is not None:
        print(f"  Reference quality: {reference_percentage:.1f}%")
        if reference_percentage < params.min_reference_percentage:
            print(f"  ⚠️ WARNING: Low reference quality (< {params.min_reference_percentage}%)")
    
    if is_inverted_curvature:
        print(f"  ⚠️ WARNING: Inverted surface curvature detected (U-shaped)")
    
    if tomato_width_mm is not None and tomato_width_mm > 0:
        print(f"  Tomato width: {tomato_width_mm:.2f} mm")
        print(f"  Grading criteria (% of tomato width):")
        print(f"    FRESH: Width ≤ 29.69% AND Depth ≤ 4.69%")
        print(f"    PROCESSING: Width ≤ 39.06% AND Depth ≤ 9.38%")
        print(f"    UNUSABLE: Exceeds processing limits")
        print(f"    CULL: Reference quality < {params.min_reference_percentage}% OR inverted curvature")
    
    if params.use_calibration_equations:
        print(f"  (Measurements shown are CALIBRATED actual values)")
    
    if defects:
        print(f"\n{'ID':<4} {'Center':<12} {'Width':<18} {'Max Depth':<12} {'Mean Depth':<12}", end='')
        if tomato_width_mm and tomato_width_mm > 0:
            print(f" {'%Width':<8} {'%Depth':<8}")
        else:
            print()
        print("-" * 90)
        
        for d in defects:
            marker = '●' if d['max_depth_mm'] > 1.0 else '○'
            width_percent = (d['width_mm'] / tomato_width_mm * 100) if tomato_width_mm and tomato_width_mm > 0 else 0
            depth_percent = (d['mean_depth_mm'] / tomato_width_mm * 100) if tomato_width_mm and tomato_width_mm > 0 else 0
            
            print(f"{marker} {d['id']:<2} {d['center_px']:>8.0f} px  "
                  f"{d['width_px']:>4} px ({d['width_mm']:>5.1f}mm)  "
                  f"{d['max_depth_mm']:>6.2f} mm    "
                  f"{d['mean_depth_mm']:>6.2f} mm", end='')
            
            if tomato_width_mm and tomato_width_mm > 0:
                print(f"    {width_percent:>5.1f}%   {depth_percent:>5.1f}%")
            else:
                print()
            
            if params.use_calibration_equations and 'width_mm_raw' in d:
                print(f"     {'':8}     Raw: ({d['width_mm_raw']:>5.1f}mm)", end='')
                if 'max_depth_mm_raw' in d:
                    print(f"  {d['max_depth_mm_raw']:>6.2f} mm    ", end='')
                if 'mean_depth_mm_raw' in d:
                    print(f"{d['mean_depth_mm_raw']:>6.2f} mm", end='')
                print()
        
        # Overall grade
        if tomato_width_mm and tomato_width_mm > 0:
            print(f"\n{'='*70}")
            print(f"  OVERALL GRADE")
            print(f"{'='*70}")
            
            # Check reference quality first
            if is_inverted_curvature:
                print(f"  🔘 Grade: CULL")
                print(f"  Reason: Inverted surface curvature (U-shaped reference)")
                print(f"  ⚠️ Detection may be unreliable - manual inspection recommended")
            elif reference_percentage is not None and reference_percentage < params.min_reference_percentage:
                print(f"  🔘 Grade: CULL")
                print(f"  Reason: Reference quality too low ({reference_percentage:.1f}%)")
                print(f"  ⚠️ Detection may be unreliable - manual inspection recommended")
            else:
                max_width = max(d['width_mm'] for d in defects)
                max_depth = max(d['mean_depth_mm'] for d in defects)
                max_width_percent = (max_width / tomato_width_mm) * 100
                max_depth_percent = (max_depth / tomato_width_mm) * 100

                print(f"max depth: {max_depth}, {tomato_width_mm}, {max_depth_percent}")
                
                print(f"  Max width: {max_width:.2f}mm ({max_width_percent:.2f}% of tomato)")
                print(f"  Max depth: {max_depth:.2f}mm ({max_depth_percent:.2f}% of tomato)")
                
                if max_width_percent <= 29.69 and max_depth_percent <= 4.69:
                    print(f"  🟢 Grade: FRESH")
                elif max_width_percent <= 39.06 and max_depth_percent <= 9.38:
                    print(f"  🟡 Grade: PROCESSING")
                else:
                    print(f"  🔴 Grade: UNUSABLE")
    else:
        print("\nNO DEFECTS DETECTED - HEALTHY TOMATO!")
        if tomato_width_mm and tomato_width_mm > 0:
            if is_inverted_curvature:
                print(f"  🔘 Grade: CULL (Inverted curvature)")
            elif reference_percentage is not None and reference_percentage < params.min_reference_percentage:
                print(f"  🔘 Grade: CULL (Low reference quality)")
            else:
                print(f"  🟢 Grade: FRESH")


# =========================
# MAIN
# =========================

def main():
    start_time = time.perf_counter()
    # print("\n" + "="*70)
    # print("  TOMATO DEFECT DETECTION - MODULAR VERSION")
    # print("  With Percentage-Based Grading")
    # print("="*70)
    
    # =================================================================
    # CONFIGURATION
    # =================================================================
    IMG_PATH = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\12-11-25\BASE_PICTURE\NATURAL\red_light\2.JPG"
    
    USE_SCALE_CORRECTION = True
    CORRECTION_FACTOR_DEPTH = 0.9622
    CORRECTION_FACTOR_WIDTH = 2.3279

    # CORRECTION_FACTOR_DEPTH = 1.0975
    # CORRECTION_FACTOR_WIDTH = 2.5197
    
    # CORRECTION_FACTOR_DEPTH = 1.1154
    # CORRECTION_FACTOR_WIDTH = 2.3949

    USE_FIXED_TOMATO_WIDTH = True
    FIXED_TOMATO_WIDTH_MM = 63.0

    # Cropping settings
    ENABLE_CROP = True
    CROP_X_START = 1700      # Start x pixel
    CROP_X_END = 4300        # End x pixel (-1 = full width)
    CROP_Y_START = 1000      # Start y pixel
    CROP_Y_END = 2900        # End y pixel (-1 = full height)
    
    # CROP_X_START = 1000
    # CROP_X_END = 2800
    # CROP_Y_START = 400
    # CROP_Y_END = 1800

    # CROP_X_START = 540
    # CROP_X_END = 1400
    # CROP_Y_START = 300
    # CROP_Y_END = 800

    LASER_COLOR = 'blue'

    FX = 4719.1
    FY = 4705.9
    CX = 3000.0
    CY = 2000.0

    # FX = 3020.2
    # FY = 2541.2
    # CX = 1920.0
    # CY = 1080.0

    # FX = 1510.1
    # FY = 1270.6
    # CX = 960.0
    # CY = 540.0
    
    # =================================================================
    
    P = Params()
    
    P.img_path = IMG_PATH
    P.laser_color = LASER_COLOR
    P.enable_crop = ENABLE_CROP
    P.crop_x_start = CROP_X_START
    P.crop_x_end = CROP_X_END
    P.crop_y_start = CROP_Y_START
    P.crop_y_end = CROP_Y_END
    P.fx = FX
    P.fy = FY
    P.cx = CX
    P.cy = CY
    P.use_scale_correction = USE_SCALE_CORRECTION
    P.correction_factor_depth = CORRECTION_FACTOR_DEPTH
    P.correction_factor_width = CORRECTION_FACTOR_WIDTH
    P.use_fixed_tomato_width = USE_FIXED_TOMATO_WIDTH
    P.fixed_tomato_width_mm = FIXED_TOMATO_WIDTH_MM

    P._recalculate_derived_params()
    
    # Load and preprocess image
    img, P = load_and_preprocess_image(P)
    
    # Extract laser stripe
    # print(f"\nExtracting {P.laser_color} laser stripe...")
    us, vs = extract_laser_color_ratio(img, P)
    
    if len(us) == 0:
        raise RuntimeError("No laser stripe detected!")
    
    # print(f"Detected: {len(us)} raw points")
    
    # Preprocess stripe
    us, vs = preprocess_stripe(us, vs, P)
    
    # Triangulate 3D points
    us, vs, depths, valid = process_triangulation(us, vs, P)
    
    # Segment tomato from floor
    depths_valid = depths[valid]
    tomato_mask_valid, floor_level = segment_tomato(depths_valid, P)
    
    # Map back to full array
    tomato_mask = np.zeros(len(us), dtype=bool)
    tomato_mask[valid] = tomato_mask_valid
    
    # print(f"Tomato points (before edge cut): {tomato_mask.sum()}")
    
    # Cut steep edges from tomato surface
    tomato_mask, cut_info = cut_tomato_edges(us, depths, tomato_mask, P)
    
    # Extract tomato width
    tomato_width_mm = None
    if cut_info and 'tomato_width_mm' in cut_info:
        tomato_width_mm = cut_info['tomato_width_mm']
    
    # print(f"Tomato points (after edge cut): {tomato_mask.sum()}")
    
    # if tomato_mask.sum() < P.min_tomato_points:
    #     print(f"WARNING: Only {tomato_mask.sum()} tomato points (minimum: {P.min_tomato_points})")
    
    # Detect defects
    defects, defect_mask, reference, edge_mask, degree_name, reference_percentage, is_inverted_curvature = detect_defects(
        us, depths, tomato_mask, P
    )
    
    # Apply calibration equations if enabled
    if P.use_calibration_equations:
        # print(f"\nApplying calibration equations...")
        defects = apply_calibration_equations(defects, P)
        # print(f"  ✓ Calibrated {len(defects)} defect measurements")
    
    # Print results with percentage-based grading
    # print_results(defects, P, tomato_width_mm, reference_percentage, is_inverted_curvature)
    
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.6f} seconds")

    # Visualize
    visualize(img, us, vs, depths, tomato_mask, defects, defect_mask, 
              reference, edge_mask, degree_name, P, cut_info, reference_percentage, is_inverted_curvature)
    
    # print(f"\n{'='*70}")
    # print("ANALYSIS COMPLETE")
    # print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()