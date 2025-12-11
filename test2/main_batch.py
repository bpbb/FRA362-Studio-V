"""
Batch processing module for tomato defect detection.
Processes all images in a folder and consolidates results.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# Import modules
from preprocess import Params, pixels_to_mm, load_and_preprocess_image
from laser_extraction import extract_laser_color_ratio, preprocess_stripe
from triangulate import process_triangulation
from segmentation import (
    segment_tomato, detect_defects, apply_calibration_equations, 
    cut_tomato_edges
)


def process_single_image(img_path, params, save_visualizations=False, output_dir=None):
    """
    Process a single image and return defect data.
    
    Args:
        img_path: Path to image file
        params: Params object (will be modified with img_path)
        save_visualizations: Whether to save visualization plots
        output_dir: Directory to save visualizations (if enabled)
    
    Returns:
        List of defect dictionaries with image metadata, or None if processing failed
    """
    try:
        # Update params with current image path
        params.img_path = str(img_path)
        
        # Load and preprocess image
        img, params = load_and_preprocess_image(params)
        
        # Extract laser stripe
        us, vs = extract_laser_color_ratio(img, params)
        
        if len(us) == 0:
            print(f"  ⚠ No laser stripe detected")
            return None
        
        # Preprocess stripe
        us, vs = preprocess_stripe(us, vs, params)
        
        # Triangulate 3D points
        us, vs, depths, valid = process_triangulation(us, vs, params)
        
        # Segment tomato from floor
        depths_valid = depths[valid]
        tomato_mask_valid, floor_level = segment_tomato(depths_valid, params)
        
        # Map back to full array
        tomato_mask = np.zeros(len(us), dtype=bool)
        tomato_mask[valid] = tomato_mask_valid
        
        # Cut steep edges
        tomato_mask, cut_info = cut_tomato_edges(us, depths, tomato_mask, params)
        
        if tomato_mask.sum() < params.min_tomato_points:
            print(f"  ⚠ Insufficient tomato points: {tomato_mask.sum()}")
            return None
        
        # Detect defects
        defects, defect_mask, reference, edge_mask, degree_name = detect_defects(
            us, depths, tomato_mask, params
        )
        
        # Apply calibration equations if enabled
        if params.use_calibration_equations:
            defects = apply_calibration_equations(defects, params)
        
        # Add image metadata to each defect
        image_name = Path(img_path).name
        for defect in defects:
            defect['image'] = image_name
            defect['image_path'] = str(img_path)
        
        # Save visualization if requested
        if save_visualizations and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            vis_filename = output_dir / f"{Path(img_path).stem}_defects.png"
            save_visualization(
                img, us, vs, depths, tomato_mask, defects, defect_mask,
                reference, edge_mask, degree_name, params, cut_info,
                str(vis_filename)
            )
        
        return defects
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def save_visualization(img, us, vs, depths, tomato_mask, defects, defect_mask, 
                      reference, edge_mask, degree_name, params, cut_info, save_path):
    """Save visualization to file instead of displaying."""
    
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
    
    # Plot floor points
    floor_edge_points = getattr(params, 'floor_edge_points', 200)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= floor_edge_points * 2:
        left_floor_indices = valid_indices[:floor_edge_points]
        right_floor_indices = valid_indices[-floor_edge_points:]
        floor_indices = np.concatenate([left_floor_indices, right_floor_indices])
        ax1.plot(us[floor_indices], vs_display[floor_indices], 'gray', 
                linewidth=2, alpha=0.6, label='Floor')
    
    # Plot tomato points
    ax1.plot(us[tomato_mask], vs_display[tomato_mask], 'lime', 
            linewidth=3, alpha=0.8, label='Tomato')
    
    # Plot defects with labels
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
        label_v = v_min - 100
        min_spacing = 50
        
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
    
    smoothing_sigma = getattr(params, 'deviation_smoothing_sigma', 15)
    depths_valid_smooth = gaussian_filter1d(depths_valid, sigma=smoothing_sigma, mode='nearest')
    
    us_mm = pixels_to_mm(us_valid, params.cx, params.pixel_size_mm)
    
    ax2.plot(us_mm[tomato_valid], depths_valid[tomato_valid], 
            'blue', linewidth=2, label='Measured (tomato)', zorder=2)
    
    # Plot floor points
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def batch_process_folder(input_folder, output_csv, params_template, 
                         save_visualizations=False, vis_output_dir=None):
    """
    Process all images in a folder and save consolidated results.
    
    Args:
        input_folder: Path to folder containing images
        output_csv: Path to output CSV file
        params_template: Template Params object (will be copied for each image)
        save_visualizations: Whether to save individual visualizations
        vis_output_dir: Directory to save visualizations
    
    Returns:
        DataFrame with all results
    """
    input_folder = Path(input_folder)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images
    image_files = [
        f for f in input_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"\n✗ No images found in {input_folder}")
        return None
    
    image_files.sort()  # Process in alphabetical order
    
    print(f"\n{'='*70}")
    print(f"  BATCH PROCESSING: {len(image_files)} images")
    print(f"{'='*70}")
    print(f"Input folder: {input_folder}")
    print(f"Output CSV: {output_csv}")
    if save_visualizations:
        print(f"Visualizations: {vis_output_dir}")
    print(f"{'='*70}\n")
    
    all_defects = []
    successful_images = 0
    failed_images = 0
    total_defects = 0
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # Create a copy of params for this image
        import copy
        params = copy.deepcopy(params_template)
        
        # Process image
        defects = process_single_image(
            img_path, params, 
            save_visualizations=save_visualizations,
            output_dir=vis_output_dir
        )
        
        if defects is not None:
            all_defects.extend(defects)
            successful_images += 1
            total_defects += len(defects)
            print(f"  ✓ Found {len(defects)} defects")
        else:
            failed_images += 1
        
        print()
    
    # Create DataFrame
    if all_defects:
        df = pd.DataFrame(all_defects)
        
        # Reorder columns for better readability
        column_order = [
            'image', 'id', 
            'width_mm', 'max_depth_mm', 'mean_depth_mm',
            'width_px', 'center_px', 'start_px', 'end_px',
            'image_path'
        ]
        
        # Add raw columns if they exist (when calibration is used)
        if 'width_mm_raw' in df.columns:
            column_order.insert(3, 'width_mm_raw')
        if 'max_depth_mm_raw' in df.columns:
            column_order.insert(4, 'max_depth_mm_raw')
        if 'mean_depth_mm_raw' in df.columns:
            column_order.insert(5, 'mean_depth_mm_raw')
        
        # Keep only columns that exist
        column_order = [col for col in column_order if col in df.columns]
        df = df[column_order]
        
        # Save to CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, float_format='%.3f')
        
        print(f"\n{'='*70}")
        print(f"  BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Successful: {successful_images}/{len(image_files)} images")
        print(f"Failed: {failed_images}/{len(image_files)} images")
        print(f"Total defects detected: {total_defects}")
        print(f"\nResults saved to: {output_csv}")
        print(f"{'='*70}\n")
        
        # Print summary statistics
        print("\nSUMMARY STATISTICS:")
        print("-" * 70)
        print(f"Average defects per image: {total_defects/successful_images:.1f}")
        print(f"\nDefect Width (mm):")
        print(f"  Mean: {df['width_mm'].mean():.2f} ± {df['width_mm'].std():.2f}")
        print(f"  Range: [{df['width_mm'].min():.2f}, {df['width_mm'].max():.2f}]")
        print(f"\nDefect Max Depth (mm):")
        print(f"  Mean: {df['max_depth_mm'].mean():.2f} ± {df['max_depth_mm'].std():.2f}")
        print(f"  Range: [{df['max_depth_mm'].min():.2f}, {df['max_depth_mm'].max():.2f}]")
        print(f"\nDefect Mean Depth (mm):")
        print(f"  Mean: {df['mean_depth_mm'].mean():.2f} ± {df['mean_depth_mm'].std():.2f}")
        print(f"  Range: [{df['mean_depth_mm'].min():.2f}, {df['mean_depth_mm'].max():.2f}]")
        print("-" * 70)
        
        return df
    else:
        print(f"\n✗ No defects detected in any images")
        return None


def main():
    """Main batch processing function."""
    print("\n" + "="*70)
    print("  TOMATO DEFECT DETECTION - BATCH PROCESSING")
    print("="*70)
    
    # =================================================================
    # CONFIGURATION - Set your parameters here
    # =================================================================
    
    # Input/Output
    INPUT_FOLDER = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\test2\26-11-25_extracted\blue"
    OUTPUT_CSV = "results.csv"
    SAVE_VISUALIZATIONS = True
    VIS_OUTPUT_DIR = "visualizations_2611_before_tuning"

    LASER_COLOR = 'blue'
    
    # Cropping settings
    ENABLE_CROP = True
    CROP_X_START = 540
    CROP_X_END = 1400
    CROP_Y_START = 300
    CROP_Y_END = 800

    FX = 1510.1
    FY = 1270.6
    CX = 960.0
    CY = 540.0

    
    # Advanced Settings
    USE_SCALE_CORRECTION = False
    CORRECTION_FACTOR_DEPTH = 0.9578
    CORRECTION_FACTOR_WIDTH = 2.2080
    
    # =================================================================
    
    # Create parameter template with system defaults
    params_template = Params()
    
    # Apply configuration
    params_template.laser_color = LASER_COLOR
    params_template.enable_crop = ENABLE_CROP
    params_template.crop_x_start = CROP_X_START
    params_template.crop_x_end = CROP_X_END
    params_template.crop_y_start = CROP_Y_START
    params_template.crop_y_end = CROP_Y_END

    params_template.fx = FX
    params_template.fy = FY
    params_template.cx = CX
    params_template.cy = CY
    
    params_template.use_scale_correction = USE_SCALE_CORRECTION
    params_template.correction_factor_depth = CORRECTION_FACTOR_DEPTH
    params_template.correction_factor_width = CORRECTION_FACTOR_WIDTH
    
    params_template._recalculate_derived_params()
    
    # Process all images
    results_df = batch_process_folder(
        input_folder=INPUT_FOLDER,
        output_csv=OUTPUT_CSV,
        params_template=params_template,
        save_visualizations=SAVE_VISUALIZATIONS,
        vis_output_dir=VIS_OUTPUT_DIR if SAVE_VISUALIZATIONS else None
    )
    
    if results_df is not None:
        print(f"\n✓ Processing complete! Results saved to {OUTPUT_CSV}")
        
        # Optional: Display first few rows
        print(f"\nFirst 10 defects:")
        print(results_df.head(10).to_string())
    else:
        print(f"\n✗ No results to save")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()