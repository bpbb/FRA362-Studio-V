"""
Batch processing module for tomato defect detection.
Processes all images in a folder with quality grading (fresh/processed).
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import time

# Import modules
from preprocess import Params, pixels_to_mm, load_and_preprocess_image
from laser_extraction import extract_laser_color_ratio, preprocess_stripe
from triangulate import process_triangulation
from segmentation import (
    segment_tomato, detect_defects, apply_calibration_equations, 
    cut_tomato_edges
)


def grade_tomato(defects, tomato_width_mm, reference_percentage=None, is_inverted_curvature=False, min_ref_threshold=30.0):
    """
    Grade tomato based on defect characteristics as percentages of tomato size.
    
    Criteria:
        - FRESH: width <= 29.69% AND depth <= 4.69%
        - PROCESSING: width <= 39.06% AND depth <= 9.38%
        - UNUSABLE: anything else
    
    Args:
        defects: List of defect dictionaries
        tomato_width_mm: Width of tomato in mm (from edge cutting)
    
    Returns:
        tuple: (grade, reason, max_width_percent, max_depth_percent)
    """
    
    if is_inverted_curvature:
        return 'cull', 'Inverted surface curvature detected (U-shaped reference)', 0.0, 0.0
    
    if reference_percentage is not None and reference_percentage < min_ref_threshold:
        return 'cull', f'Reference quality too low ({reference_percentage:.1f}% < {min_ref_threshold}%)', 0.0, 0.0
    
    if not defects or tomato_width_mm <= 0:
        return 'fresh', 'No defects detected', 0.0, 0.0
    
    # Find maximum defect percentages
    max_width_percent = 0.0
    max_depth_percent = 0.0
    worst_defect_id = 0
    
    for defect in defects:
        width_percent = (defect['width_mm'] / tomato_width_mm) * 100
        depth_percent = (defect['mean_depth_mm'] / tomato_width_mm) * 100
        
        # Track maximum percentages
        if width_percent > max_width_percent:
            max_width_percent = width_percent
        if depth_percent > max_depth_percent:
            max_depth_percent = depth_percent
            worst_defect_id = defect['id']
    
    # Apply grading criteria
    if max_width_percent <= 29.69 and max_depth_percent <= 4.69:
        return 'fresh', 'All defects within fresh grade limits', max_width_percent, max_depth_percent
    elif max_width_percent <= 39.06 and max_depth_percent <= 9.38:
        if max_width_percent > 29.69:
            return 'processing', f"Defect {worst_defect_id} width {max_width_percent:.2f}% > 29.69%", max_width_percent, max_depth_percent
        else:
            return 'processing', f"Defect {worst_defect_id} depth {max_depth_percent:.2f}% > 4.69%", max_width_percent, max_depth_percent
    else:
        if max_width_percent > 39.06:
            return 'unusable', f"Defect {worst_defect_id} width {max_width_percent:.2f}% > 39.06%", max_width_percent, max_depth_percent
        else:
            return 'unusable', f"Defect {worst_defect_id} depth {max_depth_percent:.2f}% > 9.38%", max_width_percent, max_depth_percent


def process_single_image(img_path, params):
    """
    Process a single image and return defect data with processing results.
    
    Args:
        img_path: Path to image file
        params: Params object (will be modified with img_path)
    
    Returns:
        dict with 'defects' list and processing 'data', or None if processing failed
    """
    try:
        # Update params with current image path
        params.img_path = str(img_path)
        
        # Load and preprocess image
        img, params = load_and_preprocess_image(params)
        
        # Extract laser stripe
        us, vs = extract_laser_color_ratio(img, params)
        
        if len(us) == 0:
            # print(f"  ⚠ No laser stripe detected")
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
        defects, defect_mask, reference, edge_mask, degree_name, reference_percentage, is_inverted_curvature = detect_defects(
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
        
        # Return both defects and processing data for visualization
        return {
            'defects': defects,
            'reference_percentage': reference_percentage,
            'is_inverted_curvature': is_inverted_curvature,
            'data': {
                'img': img, 'us': us, 'vs': vs, 'depths': depths,
                'tomato_mask': tomato_mask, 'defect_mask': defect_mask,
                'reference': reference, 'edge_mask': edge_mask,
                'degree_name': degree_name, 'cut_info': cut_info
            }
        }
        
    except Exception as e:
        # print(f"  ✗ Error: {e}")
        return None


def save_individual_visualization(img, us, vs, depths, tomato_mask, defects, defect_mask,
                                 reference, edge_mask, degree_name, params, cut_info, 
                                 reference_percentage, is_inverted_curvature, save_path):
    """Save full 3-plot visualization for individual image."""
    
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


def save_combined_visualization(max_depth_data, max_width_data, 
                               tomato_grade, save_path, folder_name):
    """
    Save a combined visualization showing max depth and max width images side by side.
    
    Args:
        max_depth_data: dict with image data for max depth defect
        max_width_data: dict with image data for max width defect
        tomato_grade: str, 'fresh', 'processing', or 'unusable'
        save_path: str, path to save the visualization
        folder_name: str, name of the folder (tomato identifier)
    """
    fig = plt.figure(figsize=(24, 10))
    
    # Determine grade color
    if tomato_grade == 'fresh':
        grade_color = '#90EE90'
    elif tomato_grade == 'processing':
        grade_color = '#FFD700'
    elif tomato_grade == 'cull':
        grade_color = '#808080'
    else:
        grade_color = '#FFB6C6'
    
    grade_text = f"TOMATO GRADE: {tomato_grade.upper()}"
    
    # Add main title
    fig.suptitle(f"{grade_text} | Folder: {folder_name}", 
                fontsize=20, fontweight='bold', 
                bbox=dict(boxstyle='round,pad=1', facecolor=grade_color, alpha=0.8))
    
    # Create two subplots side by side
    for idx, (data, title_prefix) in enumerate([
        (max_depth_data, f"MAX DEPTH ({max_depth_data['max_value']:.2f} mm)"),
        (max_width_data, f"MAX WIDTH ({max_width_data['max_value']:.2f} mm)")
    ]):
        ax = fig.add_subplot(1, 2, idx+1)
        
        img = data['img']
        us = data['us']
        vs = data['vs']
        depths = data['depths']
        tomato_mask = data['tomato_mask']
        defects = data['defects']
        defect_mask = data['defect_mask']
        params = data['params']
        
        # Get tomato width from params or calculate from defects
        tomato_width_mm = None
        if 'cut_info' in data:
            tomato_width_mm = data['cut_info'].get('tomato_width_mm')
        
        # Prepare display coordinates
        vs_display = vs.copy()
        if params.flip_image_vertical:
            vs_display = img.shape[0] - 1 - vs_display
        
        valid = ~np.isnan(depths)
        
        # Display image
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Plot floor points
        floor_edge_points = getattr(params, 'floor_edge_points', 200)
        valid_indices = np.where(valid)[0]
        if len(valid_indices) >= floor_edge_points * 2:
            left_floor_indices = valid_indices[:floor_edge_points]
            right_floor_indices = valid_indices[-floor_edge_points:]
            floor_indices = np.concatenate([left_floor_indices, right_floor_indices])
            ax.plot(us[floor_indices], vs_display[floor_indices], 'gray', 
                   linewidth=2, alpha=0.6, label='Floor')
        
        # Plot tomato points
        ax.plot(us[tomato_mask], vs_display[tomato_mask], 'lime', 
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
            
            # CULL grade gets gray color for all defects
            if tomato_grade == 'cull':
                color = '#808080'  # Gray - CULL
            elif tomato_width_mm and tomato_width_mm > 0:
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
            
            ax.plot([defect_center_u, defect_center_u], 
                   [label_v + 20, v_min - 5],
                   color=color, linewidth=2, alpha=0.7, zorder=4)
            
            ax.text(defect_center_u, label_v, 
                   f"{d['id']}) D = {d['mean_depth_mm']:.1f} mm, W = {d['width_mm']:.1f} mm", 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.7', facecolor=color, 
                            edgecolor='white', linewidth=3, alpha=0.95),
                   zorder=6)
        
        ax.set_title(f"{title_prefix}\nImage: {data['image_name']}", 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=11)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    # print(f"\n  💾 Saved combined visualization: {save_path}")


def batch_process_folder(input_folder, params_template, vis_output_dir=None):
    """
    Process all images in a folder (1 tomato) and give a single grade.
    
    Args:
        input_folder: Path to folder containing images from one tomato
        params_template: Template Params object (will be copied for each image)
        vis_output_dir: Directory to save visualizations (individual + combined)
    
    Returns:
        dict with processing results
    """
    input_folder = Path(input_folder)
    folder_name = input_folder.name
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images
    image_files = [
        f for f in input_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        # print(f"\n✗ No images found in {input_folder}")
        return None
    
    image_files.sort()  # Process in alphabetical order
    
    # print(f"\n{'='*70}")
    # print(f"  TOMATO QUALITY GRADING")
    # print(f"{'='*70}")
    # print(f"Folder: {folder_name}")
    # print(f"Images to process: {len(image_files)}")
    # print(f"Grading criteria (% of tomato width):")
    # print(f"  • FRESH: Width ≤ 29.69% AND Depth ≤ 4.69%")
    # print(f"  • PROCESSING: Width ≤ 39.06% AND Depth ≤ 9.38%")
    # print(f"  • UNUSABLE: Exceeds processing limits")
    # if vis_output_dir:
    #     print(f"Saving visualizations: {vis_output_dir}/")
    #     print(f"  • Individual: for each image with defects")
    #     print(f"  • Combined: max depth + max width images")
    # print(f"{'='*70}\n")
    
    # Storage for all defects from all images
    all_defects = []
    image_results = []
    
    max_depth_global = 0
    max_width_global = 0
    max_depth_image_name = None
    max_width_image_name = None
    max_depth_result = None
    max_width_result = None
    
    successful_images = 0
    failed_images = 0
    images_with_defects = 0
    
    # Track tomato width (use maximum width found across all images)
    tomato_width_mm = 0
    
    # Create individual visualization directory
    individual_vis_dir = None
    if vis_output_dir:
        individual_vis_dir = Path(vis_output_dir) / "individual"
        individual_vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for i, img_path in enumerate(image_files, 1):
        # print(f"[{i}/{len(image_files)}] Processing: {img_path.name}")
        
        # Create a copy of params for this image
        import copy
        params = copy.deepcopy(params_template)
        
        # Process image
        result = process_single_image(img_path, params)
        
        if result is not None:
            defects = result['defects']
            successful_images += 1
            
            # Get tomato width from cut_info
            cut_info = result['data'].get('cut_info')
            if cut_info and 'tomato_width_mm' in cut_info:
                image_width = cut_info['tomato_width_mm']
                if image_width > tomato_width_mm:
                    tomato_width_mm = image_width
                # print(f"  📏 Tomato width: {image_width:.1f} mm")
            
            # Store result for later use
            image_results.append({
                'image_name': img_path.name,
                'defects': defects,
                'result': result
            })
            
            # Check for defects
            if len(defects) > 0:
                images_with_defects += 1
                all_defects.extend(defects)
                
                # Save individual visualization for this image
                if individual_vis_dir:
                    vis_filename = individual_vis_dir / f"{img_path.stem}_defects.png"
                    save_individual_visualization(
                        result['data']['img'],
                        result['data']['us'],
                        result['data']['vs'],
                        result['data']['depths'],
                        result['data']['tomato_mask'],
                        defects,
                        result['data']['defect_mask'],
                        result['data']['reference'],
                        result['data']['edge_mask'],
                        result['data']['degree_name'],
                        params,
                        result['data']['cut_info'],
                        result.get('reference_percentage'), 
                        result.get('is_inverted_curvature'),
                        str(vis_filename)
                    )
                    # print(f"  💾 Saved: individual/{vis_filename.name}")
                
                # Track maximum values across ALL images
                max_depth_in_image = max(d['mean_depth_mm'] for d in defects)
                max_width_in_image = max(d['width_mm'] for d in defects)
                
                if max_depth_in_image > max_depth_global:
                    max_depth_global = max_depth_in_image
                    max_depth_image_name = img_path.name
                    max_depth_result = result
                
                if max_width_in_image > max_width_global:
                    max_width_global = max_width_in_image
                    max_width_image_name = img_path.name
                    max_width_result = result
                
                # print(f"  ✓ Found {len(defects)} defects | Max depth: {max_depth_in_image:.2f}mm | Max width: {max_width_in_image:.2f}mm")
            # else:
                # print(f"  ✓ No defects detected")
        else:
            failed_images += 1
    
    # Collect reference quality metrics from all images
    reference_percentages = [img_result['result'].get('reference_percentage', 100.0) 
                            for img_result in image_results]
    is_inverted_flags = [img_result['result'].get('is_inverted_curvature', False) 
                        for img_result in image_results]

    # Separate reliable vs unreliable images
    reliable_images = []
    unreliable_images = []

    for img_result, ref_pct, is_inverted in zip(image_results, reference_percentages, is_inverted_flags):
        is_reliable = (ref_pct >= params_template.min_reference_percentage and not is_inverted)
        
        if is_reliable:
            reliable_images.append(img_result)
        else:
            unreliable_images.append({
                'name': img_result['image_name'],
                'ref_pct': ref_pct,
                'inverted': is_inverted
            })

    # print(f"\n📊 REFERENCE QUALITY SUMMARY:")
    # print(f"  Reliable images (good quality): {len(reliable_images)}/{len(image_results)}")
    # print(f"  Unreliable images (low quality/inverted): {len(unreliable_images)}/{len(image_results)}")

    if unreliable_images:
        # print(f"\n  ⚠️ Unreliable images:")
        for img in unreliable_images:
            reason = "inverted curvature" if img['inverted'] else f"low ref quality ({img['ref_pct']:.1f}%)"
            # print(f"    • {img['name']}: {reason}")

    # Decision logic:
    # 1. If there are ANY reliable images that detected defects → use those for grading
    # 2. If ALL images are unreliable → mark as CULL
    if len(reliable_images) > 0:
        # Use defects from reliable images only
        reliable_defects = []
        for img_result in reliable_images:
            reliable_defects.extend(img_result['defects'])
        
        # print(f"\n  ✓ Using {len(reliable_defects)} defects from {len(reliable_images)} reliable images for grading")
        
        # Grade based on reliable defects only
        tomato_grade, grade_reason, max_width_percent, max_depth_percent = grade_tomato(
            reliable_defects, tomato_width_mm, 100.0, False, params_template.min_reference_percentage
        )
    else:
        # ALL images are unreliable - mark as CULL
        print(f"\n  ✗ ALL images unreliable - cannot grade defects accurately")
        tomato_grade = 'cull'
        grade_reason = 'All images have unreliable detection (low reference quality or inverted curvature)'
        max_width_percent = 0.0
        max_depth_percent = 0.0
        
    # Print results
    # print(f"\n{'='*70}")
    # print(f"  PROCESSING COMPLETE")
    # print(f"{'='*70}")
    # print(f"\n📊 PROCESSING SUMMARY:")
    # print(f"  • Successful: {successful_images}/{len(image_files)} images")
    # print(f"  • Failed: {failed_images}/{len(image_files)} images")
    # print(f"  • Images with defects: {images_with_defects}")
    # print(f"  • Total defects detected: {len(all_defects)}")
    # print(f"  • Tomato width: {tomato_width_mm:.1f} mm")
    
    # Display the final grade with percentages
    if tomato_grade == 'fresh':
        status_emoji = "🟢"
    elif tomato_grade == 'processing':
        status_emoji = "🟡"
    else:  # unusable
        status_emoji = "🔴"
    
    # print(f"\n{status_emoji} TOMATO GRADE: {tomato_grade.upper()}")
    # print(f"  Reason: {grade_reason}")
    # if len(all_defects) > 0:
    #     print(f"  Max defect width: {max_width_percent:.2f}% of tomato width")
    #     print(f"  Max defect depth: {max_depth_percent:.2f}% of tomato width")
    
    # if max_depth_image_name:
    #     print(f"\n📏 MAXIMUM VALUES ACROSS ALL IMAGES:")
    #     print(f"  • Max Depth: {max_depth_global:.2f} mm")
    #     print(f"    Found in: {max_depth_image_name}")
    #     print(f"  • Max Width: {max_width_global:.2f} mm")
    #     print(f"    Found in: {max_width_image_name}")
    
    if len(all_defects) > 0:
        # print(f"\n📈 DEFECT STATISTICS (all {len(all_defects)} defects):")
        widths = [d['width_mm'] for d in all_defects]
        depths = [d['mean_depth_mm'] for d in all_defects]
        
        # print(f"  Width (mm):")
        # print(f"    Mean: {np.mean(widths):.2f} ± {np.std(widths):.2f}")
        # print(f"    Range: [{min(widths):.2f}, {max(widths):.2f}]")
        # print(f"  Max Depth (mm):")
        # print(f"    Mean: {np.mean(depths):.2f} ± {np.std(depths):.2f}")
        # print(f"    Range: [{min(depths):.2f}, {max(depths):.2f}]")
    
    # Save combined visualization if we have defects from RELIABLE images
    if len(reliable_images) > 0 and any(len(img['defects']) > 0 for img in reliable_images) and vis_output_dir is not None:
        output_dir = Path(vis_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find max depth and width from RELIABLE images only
        max_depth_from_reliable = 0
        max_width_from_reliable = 0
        max_depth_reliable_result = None
        max_width_reliable_result = None
        max_depth_reliable_name = None
        max_width_reliable_name = None
        
        for img_result in reliable_images:
            if len(img_result['defects']) > 0:
                img_max_depth = max(d['mean_depth_mm'] for d in img_result['defects'])
                img_max_width = max(d['width_mm'] for d in img_result['defects'])
                
                if img_max_depth > max_depth_from_reliable:
                    max_depth_from_reliable = img_max_depth
                    max_depth_reliable_result = img_result['result']
                    max_depth_reliable_name = img_result['image_name']
                
                if img_max_width > max_width_from_reliable:
                    max_width_from_reliable = img_max_width
                    max_width_reliable_result = img_result['result']
                    max_width_reliable_name = img_result['image_name']
        
        # Prepare data for visualization using reliable images
        max_depth_data = {
            'img': max_depth_reliable_result['data']['img'],
            'us': max_depth_reliable_result['data']['us'],
            'vs': max_depth_reliable_result['data']['vs'],
            'depths': max_depth_reliable_result['data']['depths'],
            'tomato_mask': max_depth_reliable_result['data']['tomato_mask'],
            'defects': max_depth_reliable_result['defects'],
            'defect_mask': max_depth_reliable_result['data']['defect_mask'],
            'params': params_template,
            'cut_info': max_depth_reliable_result['data']['cut_info'],
            'image_name': max_depth_reliable_name,
            'max_value': max_depth_from_reliable
        }
        
        max_width_data = {
            'img': max_width_reliable_result['data']['img'],
            'us': max_width_reliable_result['data']['us'],
            'vs': max_width_reliable_result['data']['vs'],
            'depths': max_width_reliable_result['data']['depths'],
            'tomato_mask': max_width_reliable_result['data']['tomato_mask'],
            'defects': max_width_reliable_result['defects'],
            'defect_mask': max_width_reliable_result['data']['defect_mask'],
            'params': params_template,
            'cut_info': max_width_reliable_result['data']['cut_info'],
            'image_name': max_width_reliable_name,
            'max_value': max_width_from_reliable
        }
        
        vis_filename = output_dir / f"{folder_name}_grade_{tomato_grade}.png"
        save_combined_visualization(
            max_depth_data, max_width_data, 
            tomato_grade, str(vis_filename), folder_name
        )
    
    # print(f"{'='*70}\n")
    
    return {
        'folder_name': folder_name,
        'grade': tomato_grade,
        'grade_reason': grade_reason,
        'total_defects': len(all_defects),
        'max_depth': max_depth_global,
        'max_depth_image': max_depth_image_name,
        'max_width': max_width_global,
        'max_width_image': max_width_image_name,
        'tomato_width_mm': tomato_width_mm,
        'max_width_percent': max_width_percent,
        'max_depth_percent': max_depth_percent,
        'successful_images': successful_images,
        'failed_images': failed_images
    }


def main():
    """Main batch processing function for a single tomato (one folder)."""
    # print("\n" + "="*70)
    # print("  TOMATO DEFECT DETECTION - SINGLE TOMATO GRADING")
    # print("="*70)
    
    # =================================================================
    # CONFIGURATION - Set your parameters here
    # =================================================================
    start_time = time.perf_counter()

    # Input/Output - ONE FOLDER = ONE TOMATO
    INPUT_FOLDER = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\12-11-25\SPIN_RANDOM\ARTIFICIAL\11\no_light"
    VIS_OUTPUT_DIR = "spin_random_11_video/no_light"

    USE_SCALE_CORRECTION = True
    CORRECTION_FACTOR_DEPTH = 1.0975
    CORRECTION_FACTOR_WIDTH = 2.5197
    
    # CORRECTION_FACTOR_DEPTH = 1.1154
    # CORRECTION_FACTOR_WIDTH = 2.3949

    USE_FIXED_TOMATO_WIDTH = True
    FIXED_TOMATO_WIDTH_MM = 63.0

    # Cropping settings
    ENABLE_CROP = True
    # CROP_X_START = 1700      # Start x pixel
    # CROP_X_END = 4300        # End x pixel (-1 = full width)
    # CROP_Y_START = 1000      # Start y pixel
    # CROP_Y_END = 2900        # End y pixel (-1 = full height)
    
    # CROP_X_START = 1000
    # CROP_X_END = 2800
    # CROP_Y_START = 400
    # CROP_Y_END = 1800

    CROP_X_START = 540
    CROP_X_END = 1400
    CROP_Y_START = 300
    CROP_Y_END = 800

    LASER_COLOR = 'blue'

    # FX = 4719.1
    # FY = 4705.9
    # CX = 3000.0
    # CY = 2000.0

    # FX = 3020.2
    # FY = 2541.2
    # CX = 1920.0
    # CY = 1080.0

    FX = 1510.1
    FY = 1270.6
    CX = 960.0
    CY = 540.0
    
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
    
    params_template.use_fixed_tomato_width = USE_FIXED_TOMATO_WIDTH
    params_template.fixed_tomato_width_mm = FIXED_TOMATO_WIDTH_MM

    params_template._recalculate_derived_params()
    
    # Process all images in the folder (1 tomato)
    result = batch_process_folder(
        input_folder=INPUT_FOLDER,
        params_template=params_template,
        vis_output_dir=VIS_OUTPUT_DIR
    )

    end_time = time.perf_counter()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.6f} seconds")
    
    # if result is not None:
    #     print(f"\n✅ FINAL RESULT:")
    #     print(f"  Tomato: {result['folder_name']}")
    #     print(f"  Grade: {result['grade'].upper()}")
    #     print(f"  Tomato width: {result['tomato_width_mm']:.1f} mm")
    #     print(f"  Total defects: {result['total_defects']}")
    #     if result['total_defects'] > 0:
    #         print(f"  Max depth: {result['max_depth']:.2f}mm ({result['max_depth_percent']:.2f}% of width)")
    #         print(f"    Image: {result['max_depth_image']}")
    #         print(f"  Max width: {result['max_width']:.2f}mm ({result['max_width_percent']:.2f}% of width)")
    #         print(f"    Image: {result['max_width_image']}")
    # else:
    #     print(f"\n✗ No results to process")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()