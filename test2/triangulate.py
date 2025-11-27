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


def detrend_floor_tilt(us, depths, method="linear", floor_edge_points=200):
    """
    Remove floor tilt by fitting and subtracting a trend line.
    
    Uses far left and far right points which are guaranteed to be floor.
    
    Args:
        us: X coordinates (pixels)
        depths: Depth values (mm)
        method: Detrending method ("linear" or "polynomial")
        floor_edge_points: Number of points from each edge to use as floor
    
    Returns:
        depths_detrended: Corrected depths
        trend: The fitted trend line
        floor_baseline: Floor level after detrending
    """
    n = len(depths)
    
    # Use far left and far right points - these are always floor
    edge_points = min(floor_edge_points, n // 4)  # Don't use more than 25% from each side
    
    if n < edge_points * 2 + 50:
        print(f"\nDetrending floor tilt: Not enough points ({n}), skipping")
        return depths, np.zeros_like(depths), np.median(depths)
    
    # Get floor points from edges
    left_indices = np.arange(edge_points)
    right_indices = np.arange(n - edge_points, n)
    floor_indices = np.concatenate([left_indices, right_indices])
    
    floor_us = us[floor_indices]
    floor_depths = depths[floor_indices]
    
    print(f"\nDetrending floor tilt:")
    print(f"   Using {edge_points} points from each edge (total: {edge_points*2})")
    
    if method == "linear":
        coeffs = np.polyfit(floor_us, floor_depths, deg=1)
        trend = np.polyval(coeffs, us)
        
        tilt_angle = np.rad2deg(np.arctan(coeffs[0]))
        print(f"   Linear fit: slope={coeffs[0]:.6f} mm/px ({tilt_angle:.3f} deg)")
        print(f"   Tilt across image: {coeffs[0] * n:.2f} mm")
        
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
    
    # Verify detrending worked
    floor_depths_after = depths_detrended[floor_indices]
    left_floor_after = np.median(floor_depths_after[:edge_points])
    right_floor_after = np.median(floor_depths_after[-edge_points:])
    
    correction_range = trend.max() - trend.min()
    print(f"   Removed tilt: {correction_range:.2f} mm range")
    print(f"   Floor level: {floor_median:.1f} mm")
    print(f"   After detrend - Left floor: {left_floor_after:.2f}mm, Right floor: {right_floor_after:.2f}mm")
    print(f"   Difference: {abs(left_floor_after - right_floor_after):.2f}mm (should be ~0)")
    
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
    
    # Get floor_edge_points from params
    floor_edge_points = getattr(params, 'floor_edge_points', 200)
    
    # Detrend floor tilt
    depths_detrended, floor_trend, floor_baseline = detrend_floor_tilt(
        us_valid, depths_valid, method=detrend_method, floor_edge_points=floor_edge_points
    )
    
    # Put detrended depths back into main array
    depths[valid] = depths_detrended
    
    return us, vs, depths, valid