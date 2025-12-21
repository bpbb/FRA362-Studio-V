# Tomatoes’ Crack Size Sorter

## Project Overview

This project is part of **FRA362 Robotics Studio V** at the Institute of Field Robotics, King Mongkut's University of Technology Thonburi. Our team addressed the critical issue of **tomato loss in the agricultural supply chain**, a significant contributor to greenhouse gas emissions.

### The Problem

Tomatoes are the world's highest-volume fruit crop, yet they suffer massive losses throughout the supply chain:

- **Out of 100 harvested tomatoes:**
  - 27 lost during harvesting
  - 10 lost in packaging/transportation
  - 6 lost in storage & quality control
  - 19 lost as household waste
  - **Only 41 reach consumers**

A key contributor to this waste is **inefficient quality grading**. Currently, all tomatoes enter the "fresh market" processing line first, even those suitable only for processing. This creates:
- Delays in identifying processing-grade tomatoes
- Quality degradation while waiting for re-routing
- Unnecessary rejection of tomatoes that could be processed
- Increased overall loss rates

### Our Solution

We developed an **automated laser triangulation-based defect detection system** that grades tomatoes in real-time as they pass through the sorting line, enabling immediate routing to fresh or processing streams.

---

## Technical Approach

### System Architecture

**Hardware Setup:**
- Laser line projector
- 1080p camera @ 100fps

**Processing Pipeline:**

1. **Image Preprocessing & Laser Extraction**
   - Camera undistortion and cropping
   - Color ratio-based laser extraction
   - Multi-stage outlier removal
   - Savitzky-Golay smoothing

2. **3D Reconstruction**
   - Ray-plane intersection triangulation
   - Floor plane estimation and detrending
   - Automatic gap detection for tomato segmentation

3. **Edge Cutting & Width Measurement**
   - Slope-based edge detection
   - Safety margin application
   - Auto-calculated or fixed tomato width

4. **Defect Detection with Quality Validation**
   - Two-pass reference surface creation:
     - Rough reference from outer 25% edges
     - Refined reference using healthy points only
   - **Reference Quality Validation:**
     - Reference percentage
     - Surface curvature check
   - Deviation-based defect identification

5. **Percentage-Based Grading**
   - Size-independent metrics (width% and depth% relative to tomato width)
   - **Four-tier grading system:**
     - 🟢 **FRESH**: Width ≤29.69%, Depth ≤4.69%
     - 🟡 **PROCESSING**: Width ≤39.06%, Depth ≤9.38%
     - 🔴 **UNUSABLE**: Exceeds processing limits
     - ⚪ **CULL**: Unreliable detection (ref% <30% OR inverted curvature)

---

## Experimental Validation

### Parameter Optimization Experiment

**Objective:** Determine optimal laser color and lighting configuration

**Training Set:** 15 artificial cracks (2-6mm depth, 6.5-28mm width)

**Results:**
- **Blue Laser + No Light**: Best for depth (RMSE 0.24mm)
- **Blue Laser + Red Light**: Best for width (RMSE 1.56mm)
- Green laser showed higher errors across all metrics

### Dataset Verification

**Test Set:** 
- 15 artificial cracks
- 7 natural cracks

**Key Findings:**
- Artificial cracks: RMSE 0.96mm (depth), 2.42mm (width)
- Natural cracks: RMSE 1.22mm (depth), 5.95mm (width)
- Higher errors in natural cracks due to:
  - Non-perpendicular intersection with laser line
  - Irregular geometry and depth variation
  - Less defined edges

### Video Format Experiment

**Objective:** Optimize video capture settings for rolling tomatoes

**Comparison:** 1080p@100fps vs 4K@30fps

**Winner: 1080p @ 100fps**
- Reduced motion blur
- Better temporal resolution for fast-moving objects
- Clearer laser line extraction
- Faster processing

**Rolling Methods:**
- **Intend Rolling:** Controlled positioning (better width accuracy)
- **Random Rolling:** Real-world simulation (better depth accuracy)

**Video Verification Results:**
- Intend Spin: RMSE 0.89mm (depth), 2.15mm (width)
- Random Spin: RMSE 0.85mm (depth), 2.29mm (width)
- Blue Laser + No Light consistently outperformed Red Light

---

## Impact

This system addresses tomato loss by:
- **Early detection** of defects before quality degradation
- **Accurate routing** to fresh vs. processing streams
- **Reduced waste** from delayed quality decisions
- **Scalable automation** for high-throughput sorting lines

By implementing proper grading at the source, we can reduce unnecessary losses in both fresh and processing streams, contributing to more sustainable food systems.