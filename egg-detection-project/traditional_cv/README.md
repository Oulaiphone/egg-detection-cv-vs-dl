# Traditional Computer Vision - Egg Detection

This folder contains the traditional computer vision implementation for egg detection using OpenCV.

## Approach

The traditional pipeline uses handcrafted feature extraction techniques:

1. **BGR to HSV Conversion** - Convert color space for better color-based segmentation
2. **Channel Splitting** - Separate image into individual channels
3. **Morphological Closing** - Remove noise and fill small holes
4. **Distance Transform** - Calculate distance to nearest background pixel
5. **Template Creation** - Create egg template for matching
6. **Template Matching** - Find egg-like shapes in the image
7. **Local Maxima Detection** - Find centers of detected eggs
8. **Contour Finding** - Draw bounding contours around eggs

## Files

```
traditional_cv/
├── README.md           # This file
├── egg_detection.py    # Main detection script

```

## Usage (Original)

```python
# Note: This code was written for Python 3.6.10 and OpenCV 4.x in 2020

import cv2
from egg_detection import EggDetector

detector = EggDetector()
result = detector.detect(image_path)
```

## Dependencies

- Python 3.6.10
- OpenCV 4.x
- NumPy
- PyQt5 (for GUI)

## Results

- **Precision:** 99%
- **Test:** 130 eggs detected from conveyor belt video

## Limitations

- Requires controlled lighting conditions
- Works best with uniform background
- Manual feature engineering required
- Limited adaptability to new environments

## Reference

This implementation was inspired by:
- [Counting Eggs in OpenCV](https://ivanursul.com/counting-eggs-in-opencv) by Ivan Ursul

---

*Part of the Egg Detection comparative study - STEP Program 2020*
