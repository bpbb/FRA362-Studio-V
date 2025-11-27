"""
Triangulation module for tomato defect detection.
Handles 3D point triangulation and floor tilt correction.
"""

import numpy as np


def triangulate(us, vs, params):
    """
    Triangulate 3D points from 2D laser stripe using plane intersection.
    
    Args:
        us, vs: Pixel coordinates of laser stripe
        params: Params object with camera intrinsics and plane parameters
    
    Returns:
        depths: Array of Z depths (mm)
    """
    ones = np.ones_like(us)
    uv1 = np.stack([us, vs, ones], axis=1)
    
    Kinv = np.linalg.inv(params.K)
    rays = (Kinv @ uv1.T).T
    rays /= (np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12)
    
    depths = []
    for ray in rays:
        denom = np.dot(params.plane_n, ray)
        if abs(denom) > 1e-9:
            t = -params.plane_d / denom
            if t > 0:
                depths.append(ray[2] * t)
                continue
        depths.append(np.nan)
    
    return np.array(depths)


def detrend_floor_tilt(us, depths, method="linear"):
    """
    Remove floor tilt by fitting and subtracting a trend line.
    
    Methods:
        "linear": Fit straight line to edges (fast, good for small tilt)
        "polynomial": Fit polynomial to edges (better for curved floors)
        "robust": RANSAC-based fit (best for noisy data)
    
    Args:
        us: X coordinates (pixels)
        depths: Depth values (mm)
        method: Detrending method
    
    Returns:
        depths_detrended: Corrected depths
        trend: The fitted trend line
        floor_baseline: Floor level after detrending
    """
    n = len(depths)
    edge_size = max(50, int(n * 0.1))  # Use 10% from each edge
    
    # Extract floor points (edges assumed to be floor)
    floor_indices = np.concatenate([
        np.arange(edge_size),
        np.arange(n - edge_size, n)
    ])
    floor_us = us[floor_indices]
    floor_depths = depths[floor_indices]
    
    print(f"\nDetrending floor tilt:")
    print(f"   Using {len(floor_indices)} floor points from edges")
    
    if method == "linear":
        coeffs = np.polyfit(floor_us, floor_depths, deg=1)
        trend = np.polyval(coeffs, us)
        
        tilt_angle = np.rad2deg(np.arctan(coeffs[0]))
        print(f"   Linear fit: slope={coeffs[0]:.6f} mm/px ({tilt_angle:.3f} deg)")
        print(f"   Tilt across image: {coeffs[0] * n:.1f} mm")
        
    elif method == "polynomial":
        coeffs = np.polyfit(floor_us, floor_depths, deg=2)
        trend = np.polyval(coeffs, us)
        print(f"   Polynomial fit (degree 2)")
        
    elif method == "robust":
        from sklearn.linear_model import RANSACRegressor
        
        ransac = RANSACRegressor(random_state=42)
        ransac.fit(floor_us.reshape(-1, 1), floor_depths)
        trend = ransac.predict(us.reshape(-1, 1))
        print(f"   RANSAC robust fit")
    
    # Subtract the trend (normalize to median floor level)
    floor_median = np.median(floor_depths)
    depths_detrended = depths - trend + floor_median
    
    correction_range = trend.max() - trend.min()
    print(f"   Removed tilt: {correction_range:.1f} mm range")
    print(f"   New floor level: {floor_median:.1f} mm (horizontal)")
    
    return depths_detrended, trend, floor_median


def process_triangulation(us, vs, params, detrend_method="linear"):
    """
    Full triangulation pipeline with floor detrending.
    
    Args:
        us, vs: Stripe coordinates
        params: Params object
        detrend_method: Method for floor tilt correction
    
    Returns:
        us, vs, depths: Full arrays (with NaN for invalid)
        valid: Boolean mask of valid points
    """
    print(f"\nTriangulating 3D points...")
    depths = triangulate(us, vs, params)
    valid = ~np.isnan(depths)
    print(f"Valid 3D points: {valid.sum()}")
    
    # Extract valid points
    us_valid = us[valid]
    vs_valid = vs[valid]
    depths_valid = depths[valid]
    
    # Detrend floor tilt
    depths_detrended, floor_trend, floor_baseline = detrend_floor_tilt(
        us_valid, depths_valid, method=detrend_method
    )
    
    # Put detrended depths back into main array
    depths[valid] = depths_detrended
    
    return us, vs, depths, valid