"""
Preprocessing module for tomato defect detection.
Handles parameters, image loading, undistortion, and cropping.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Params:
    """Configuration parameters for tomato defect detection system."""
    
    # Calibration
    camera_angle_from_laser_deg: float = 30.0
    cam_to_object_distance_mm: float = 200.0
    
    # Scale correction
    use_scale_correction: bool = True
    correction_factor_width: float = 0.25
    correction_factor_depth: float = 2.105
    
    # Calibration equations
    use_calibration_equations: bool = False
    calibration_width_coeffs: list = field(default_factory=lambda: [-0.00759153275256077, 0.9435466942946986, -37.40261711439749, 481.9594021049923])
    calibration_depth_coeffs: list = field(default_factory=lambda: [0.7685120302546002, -16.55315145092963, 114.58591024625971, -249.04535429072686])
    
    # Image
    img_path: str = r"C:\fibo\3rd year_1st semester\studio\FRA362-Studio-V\26-11-Tomato\7M300011.JPG"
    
    # Cropping
    enable_crop: bool = True
    crop_x_start: int = 1500      # Start x pixel
    crop_x_end: int = 4400        # End x pixel (-1 = full width)
    crop_y_start: int = 1200      # Start y pixel
    crop_y_end: int = 3000        # End y pixel (-1 = full height)
    
    # Camera intrinsics
    fx: float = 4719.0
    fy: float = 4705.0
    cx: float = 3000.0
    cy: float = 2000.0
    dist: np.ndarray = field(default_factory=lambda: np.array([0,0,0,0,0], dtype=np.float64))
    flip_image_vertical: bool = True
    
    # Laser Extraction
    laser_color: str = "blue"
    bandpass_kernel: int = 9
    subpixel_halfwidth: int = 3
    color_ratio_threshold: float = 0.49
    min_val_fraction: float = 0.24

    # Initial lowpass filtering
    enable_initial_lowpass: bool = True
    lowpass_method: str = "savgol"
    lowpass_window: int = 11
    
    # Outlier removal
    enable_outlier_removal: bool = True
    outlier_method: str = "improved"
    
    # Stripe smoothing
    stripe_smoothing_sigma: float = 2.0
    
    # Segmentation
    auto_calculate_gap: bool = True
    tomato_floor_gap_mm: float = 35.0
    min_tomato_points: int = 80
    
    # Edge-Poly Reference
    edge_region_percent: float = 25.0
    edge_poly_degree: int = 2
    use_circular_reference: bool = True
    defect_adjacent_points: int = 50
    
    # Defect Detection
    defect_threshold_mm: float = 3.0
    min_defect_width_px: int = 10
    edge_exclude_percent: float = 10.0

    # Smoothing control
    depth_smoothing_sigma: float = 5.0
    use_median_prefilter: bool = True
    median_window_size: int = 5
    
    # Clustering
    cluster_defects: bool = True
    cluster_max_gap_pixels: int = 10
    
    # Edge Cutting (slope-based)
    enable_edge_cutting: bool = True
    edge_slope_threshold: float = 0.5      # mm/px - slopes above this are "steep"
    edge_min_flat_points: int = 20         # Consecutive flat points to confirm surface
    edge_smoothing_window: int = 5         # Smoothing before slope calculation
    
    # Healthy Point Validation (for reference creation)
    healthy_window_size: int = 15          # Buffer points for boundary detection
    deviation_smoothing_sigma: int = 15    # Smoothing sigma for deviation curve (higher = smoother)
    
    # Output
    out_csv: str = "defects_combined.csv"
    out_plot: str = "defect_detection_combined.png"
    
    # Computed fields
    K: np.ndarray = field(init=False)
    plane_n: np.ndarray = field(init=False)
    plane_d: float = field(init=False)
    pixel_size_mm: float = field(init=False)
    
    def __post_init__(self):
        self.K = np.array([[self.fx, 0, self.cx], 
                          [0, self.fy, self.cy], 
                          [0, 0, 1]], dtype=np.float64)
        theta = np.deg2rad(self.camera_angle_from_laser_deg)
        self.plane_n = np.array([0, np.cos(theta), np.sin(theta)])
        self.plane_n /= np.linalg.norm(self.plane_n)
        
        base_plane_d = -self.cam_to_object_distance_mm * np.sin(theta)
        
        if self.use_scale_correction:
            self.plane_d = base_plane_d * self.correction_factor_depth
        else:
            self.plane_d = base_plane_d
        
        avg_z = abs(self.plane_d)
        base_pixel_size = avg_z / self.fx
        
        if self.use_scale_correction:
            self.pixel_size_mm = base_pixel_size * self.correction_factor_width
        else:
            self.pixel_size_mm = base_pixel_size


def pixels_to_mm(pixels, center_pixel, pixel_size_mm):
    """Convert pixel coordinates to mm relative to center."""
    return (pixels - center_pixel) * pixel_size_mm


def undistort(image, K, dist):
    """Undistort image using camera calibration."""
    if np.allclose(dist, 0):
        return image, K
    h, w = image.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
    und = cv2.undistort(image, K, dist, None, newK)
    return und, newK


def crop_image(image, params):
    """
    Crop image to region of interest.
    
    Args:
        image: Input BGR image
        params: Params object with crop settings
    
    Returns:
        Cropped image and updated params with adjusted cx, cy
    """
    if not params.enable_crop:
        return image, params
    
    h, w = image.shape[:2]
    
    # Handle default values
    x_start = params.crop_x_start
    x_end = params.crop_x_end if params.crop_x_end > 0 else w
    y_start = params.crop_y_start
    y_end = params.crop_y_end if params.crop_y_end > 0 else h
    
    # Validate bounds
    x_start = max(0, min(x_start, w-1))
    x_end = max(x_start+1, min(x_end, w))
    y_start = max(0, min(y_start, h-1))
    y_end = max(y_start+1, min(y_end, h))
    
    # Crop image
    cropped = image[y_start:y_end, x_start:x_end].copy()
    
    # Adjust camera center for cropped region
    params.cx = params.cx - x_start
    params.cy = params.cy - y_start
    
    # Update K matrix
    params.K = np.array([[params.fx, 0, params.cx], 
                        [0, params.fy, params.cy], 
                        [0, 0, 1]], dtype=np.float64)
    
    print(f"Cropped image: {w}x{h} -> {cropped.shape[1]}x{cropped.shape[0]}")
    print(f"   Region: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}]")
    print(f"   Adjusted center: cx={params.cx:.1f}, cy={params.cy:.1f}")
    
    return cropped, params


def load_and_preprocess_image(params):
    """
    Load image and apply preprocessing (undistort, crop).
    
    Args:
        params: Params object
    
    Returns:
        Preprocessed image and updated params
    """
    print(f"\nLoading: {params.img_path}")
    img = cv2.imread(params.img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {params.img_path}")
    print(f"Loaded: {img.shape[1]}x{img.shape[0]}")
    
    # Undistort
    img_u, K = undistort(img, params.K, params.dist)
    params.K = K
    
    # Crop if enabled
    img_cropped, params = crop_image(img_u, params)
    
    return img_cropped, params