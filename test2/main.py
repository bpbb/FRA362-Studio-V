"""
Main module for tomato defect detection.
Orchestrates the full pipeline and provides visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
              reference, edge_mask, degree_name, params, cut_info=None):
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
    
    # Plot floor points (gray) - show the actual floor points used for detrending
    # These are the far left and far right points
    floor_edge_points = getattr(params, 'floor_edge_points', 200)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= floor_edge_points * 2:
        left_floor_indices = valid_indices[:floor_edge_points]
        right_floor_indices = valid_indices[-floor_edge_points:]
        floor_indices = np.concatenate([left_floor_indices, right_floor_indices])
        
        ax1.plot(us[floor_indices], vs_display[floor_indices], 'gray', linewidth=2, alpha=0.6, label='Floor')
    
    # Plot tomato points (lime green)
    ax1.plot(us[tomato_mask], vs_display[tomato_mask], 'lime', linewidth=3, alpha=0.8, label='Tomato')
    
    used_positions = []
    for d in defects:
        mask = defect_mask & (us >= d['start_px']) & (us <= d['end_px'])
        if not mask.any():
            continue
        
        us_def = us[mask]
        vs_def = vs_display[mask]
        v_min = vs_def.min()
        
        if d['max_depth_mm'] > 10.0:
            color = '#FF0000'
        elif d['max_depth_mm'] > 5.0:
            color = '#FFA500'
        else:
            color = '#FFD700'
        
        defect_center_u = d['center_px']
        
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
        
        ax1.plot([defect_center_u, defect_center_u], 
                [label_v + 80, v_min - 5],
                color=color, linewidth=2, alpha=0.7, zorder=4)
        
        ax1.text(defect_center_u, label_v, 
                f"{d['id']}) D = {d['mean_depth_mm']:.1f} mm, W = {d['width_mm']:.1f} mm", 
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.7', facecolor=color, 
                         edgecolor='white', linewidth=3, alpha=0.95),
                zorder=6)
    
    ax1.set_title(f"Detected Defects: {len(defects)}", fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.axis('off')
    
    # ============ Plot 2: Depth Profile ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    us_valid = us[valid]
    depths_valid = depths[valid]
    tomato_valid = tomato_mask[valid]
    edge_valid = edge_mask[valid] if edge_mask is not None else np.zeros(valid.sum(), dtype=bool)
    
    # Smooth depths for display (to match reference fitting)
    smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
    depths_valid_smooth = gaussian_filter1d(depths_valid, sigma=smoothing_sigma, mode='nearest')
    
    us_mm = pixels_to_mm(us_valid, params.cx, params.pixel_size_mm)
    
    ax2.plot(us_mm[tomato_valid], depths_valid[tomato_valid], 
            'blue', linewidth=2, label='Measured (tomato)', zorder=2)
    
    # Plot floor points (same as Plot 1 - far left and far right)
    floor_edge_points = getattr(params, 'floor_edge_points', 200)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= floor_edge_points * 2:
        left_floor_indices = valid_indices[:floor_edge_points]
        right_floor_indices = valid_indices[-floor_edge_points:]
        floor_indices = np.concatenate([left_floor_indices, right_floor_indices])
        
        us_floor_mm = pixels_to_mm(us[floor_indices], params.cx, params.pixel_size_mm)
        ax2.plot(us_floor_mm, depths[floor_indices],
                'gray', linewidth=1, alpha=0.5, label='Floor', zorder=1)
    
    # Determine reference method label
    if edge_mask is not None and edge_mask.any():
        edge_indices = np.where(edge_mask[valid] & tomato_valid)[0]
        tomato_indices = np.where(tomato_valid)[0]
        
        if len(edge_indices) > 0 and len(tomato_indices) > 0:
            tomato_start = tomato_indices[0]
            tomato_end = tomato_indices[-1]
            tomato_length = tomato_end - tomato_start
            
            edge_in_center_count = 0
            for idx in edge_indices:
                relative_pos = (idx - tomato_start) / tomato_length
                if 0.2 < relative_pos < 0.8:
                    edge_in_center_count += 1
            
            if edge_in_center_count > len(edge_indices) * 0.5:
                if degree_name == "circular arc":
                    title_text = f'Circular arc fit to defect-adjacent regions'
                else:
                    title_text = f'Polynomial ({degree_name}) fit to defect-adjacent regions'
                ref_label = 'Local reference'
            else:
                if degree_name == "circular arc":
                    title_text = f'Circular arc fit to tomato edges'
                else:
                    title_text = f'Polynomial ({degree_name}) fit to tomato edges'
                ref_label = 'Edge regions'
        else:
            title_text = f'{degree_name.title()} fit'
            ref_label = 'Reference points'
    else:
        title_text = f'{degree_name.title()} fit'
        ref_label = 'Reference points'
    
    if edge_valid.any():
        ax2.scatter(us_mm[edge_valid & tomato_valid], depths_valid_smooth[edge_valid & tomato_valid],
                   c='green', s=10, alpha=0.7, marker='s', label=ref_label, zorder=3)
    
    ref_valid = reference[valid]
    ref_mask = ~np.isnan(ref_valid)
    ax2.plot(us_mm[ref_mask], ref_valid[ref_mask],
            'orange', linewidth=2, linestyle='--', label='Reference', zorder=4)
    
    # Plot defect markers on smoothed depths (to match reference curve)
    for d in defects:
        mask = defect_mask[valid] & (us_valid >= d['start_px']) & (us_valid <= d['end_px'])
        if mask.any():
            color = '#FF0000' if d['max_depth_mm'] > 10.0 else '#FFA500' if d['max_depth_mm'] > 5.0 else '#FFD700'
            ax2.scatter(us_mm[mask], depths_valid_smooth[mask], c=color, s=20, 
                       marker='v', edgecolors='darkred', linewidths=0.5, zorder=5)
    
    # Draw edge cut lines on depth profile
    if cut_info is not None and cut_info.get('left_u') is not None:
        left_u_mm = pixels_to_mm(cut_info['left_u'], params.cx, params.pixel_size_mm)
        right_u_mm = pixels_to_mm(cut_info['right_u'], params.cx, params.pixel_size_mm)
        ax2.axvline(x=left_u_mm, color='red', linewidth=2, linestyle='--', 
                   alpha=0.4, label='Edge cuts')
        ax2.axvline(x=right_u_mm, color='red', linewidth=2, linestyle='--', alpha=0.4)
    
    ax2.set_xlabel('Position (mm)', fontsize=11)
    ax2.set_ylabel('Depth Z (mm)', fontsize=11)
    ax2.set_title(title_text, fontsize=12, fontweight='bold')
    ax2.legend(fontsize=7, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    if tomato_valid.any():
        # Set Y limits (depth) based on tomato region
        tomato_depths = depths_valid[tomato_valid]
        depth_min = tomato_depths.min()
        depth_max = tomato_depths.max()
        depth_range = depth_max - depth_min
        
        margin_top = depth_range * 0.60
        margin_bottom = depth_range * 0.15
        y_min = depth_min - margin_top
        y_max = depth_max + margin_bottom
        
        ax2.set_ylim(y_max, y_min)
        
        # Set X limits to zoom in on tomato region only
        tomato_us_mm = us_mm[tomato_valid]
        x_min = tomato_us_mm.min()
        x_max = tomato_us_mm.max()
        x_range = x_max - x_min
        x_margin = x_range * 0.1  # 10% margin on each side
        
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
            # Plot original deviation (light blue, thin)
            ax3.plot(us_tomato[valid_dev], dev_tomato[valid_dev], 
                    'blue', linewidth=1.5, alpha=0.5, label='Deviation (raw)')
            
            # Plot smoothed deviation (dark blue, thick)
            smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
            dev_smooth = gaussian_filter1d(dev_tomato[valid_dev], sigma=smoothing_sigma, mode='nearest')
            ax3.plot(us_tomato[valid_dev], dev_smooth, 
                    'darkblue', linewidth=2, label='Deviation (smoothed)')
            
            ax3.axhline(0, color='green', linewidth=2, alpha=0.6, label='Baseline')
            ax3.axhline(params.defect_threshold_mm, color='red', linewidth=2, 
                       linestyle='--', label=f'Threshold ({params.defect_threshold_mm}mm)')
            
            # Show reference points on the SMOOTHED deviation line (green squares)
            if edge_mask is not None:
                edge_tomato = edge_mask[tomato_mask] & valid_dev
                # Get the smoothed deviation values for reference points
                # We need to map edge_tomato indices to dev_smooth indices
                edge_indices_in_valid = np.where(edge_tomato[valid_dev])[0]
                ax3.scatter(us_tomato[valid_dev][edge_indices_in_valid], 
                          dev_smooth[edge_indices_in_valid],
                          c='green', s=40, alpha=0.5, marker='s', zorder=5,
                          label='Reference points')
            
            # Fill defect regions
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


def print_results(defects, params):
    """Print detection results to console."""
    print(f"\n{'='*70}")
    print(f"  RESULTS: {len(defects)} DEFECTS DETECTED")
    print(f"{'='*70}")
    
    if params.use_calibration_equations:
        print(f"  (Measurements shown are CALIBRATED actual values)")
    
    if defects:
        print(f"\n{'ID':<4} {'Center':<12} {'Width':<18} {'Max Depth':<12} {'Mean Depth':<12}")
        print("-" * 70)
        for d in defects:
            marker = '●' if d['max_depth_mm'] > 1.0 else '○'
            print(f"{marker} {d['id']:<2} {d['center_px']:>8.0f} px  "
                  f"{d['width_px']:>4} px ({d['width_mm']:>5.1f}mm)  "
                  f"{d['max_depth_mm']:>6.2f} mm    "
                  f"{d['mean_depth_mm']:>6.2f} mm")
            
            if params.use_calibration_equations and 'width_mm_raw' in d:
                print(f"     {'':8}     Raw: ({d['width_mm_raw']:>5.1f}mm)", end='')
                if 'max_depth_mm_raw' in d:
                    print(f"  {d['max_depth_mm_raw']:>6.2f} mm    ", end='')
                if 'mean_depth_mm_raw' in d:
                    print(f"{d['mean_depth_mm_raw']:>6.2f} mm", end='')
                print()
    else:
        print("\nNO DEFECTS DETECTED - HEALTHY TOMATO!")


# =========================
# MAIN
# =========================

def main():
    print("\n" + "="*70)
    print("  TOMATO DEFECT DETECTION - MODULAR VERSION")
    print("  With Enhanced Outlier Removal & Lowpass Filtering")
    print("="*70)
    
    # Create parameters
    P = Params()
    
    # Load and preprocess image
    img, P = load_and_preprocess_image(P)
    
    # Extract laser stripe
    print(f"\nExtracting {P.laser_color} laser stripe...")
    us, vs = extract_laser_color_ratio(img, P)
    
    if len(us) == 0:
        raise RuntimeError("No laser stripe detected!")
    
    print(f"Detected: {len(us)} raw points")
    
    # Preprocess stripe (lowpass, outlier removal, smoothing)
    us, vs = preprocess_stripe(us, vs, P)
    
    # Triangulate 3D points (includes initial detrending)
    us, vs, depths, valid = process_triangulation(us, vs, P)
    
    # Segment tomato from floor
    depths_valid = depths[valid]
    tomato_mask_valid, floor_level = segment_tomato(depths_valid, P)
    
    # Map back to full array
    tomato_mask = np.zeros(len(us), dtype=bool)
    tomato_mask[valid] = tomato_mask_valid
    
    print(f"Tomato points (before edge cut): {tomato_mask.sum()}")
    
    # Cut steep edges from tomato surface
    tomato_mask, cut_info = cut_tomato_edges(us, depths, tomato_mask, P)
    
    print(f"Tomato points (after edge cut): {tomato_mask.sum()}")
    
    if tomato_mask.sum() < P.min_tomato_points:
        print(f"WARNING: Only {tomato_mask.sum()} tomato points (minimum: {P.min_tomato_points})")
    
    # Detect defects
    defects, defect_mask, reference, edge_mask, degree_name = detect_defects(
        us, depths, tomato_mask, P
    )
    
    # Apply calibration equations if enabled
    if P.use_calibration_equations:
        print(f"\nApplying calibration equations...")
        defects = apply_calibration_equations(defects, P)
        print(f"  ✓ Calibrated {len(defects)} defect measurements")
    
    # Print results
    print_results(defects, P)
    
    # Save results (uncomment if needed)
    # save_results(us, vs, depths, tomato_mask, reference, defects, defect_mask, P)
    
    # Visualize
    visualize(img, us, vs, depths, tomato_mask, defects, defect_mask, 
              reference, edge_mask, degree_name, P, cut_info)
    
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