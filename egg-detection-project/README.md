# 🥚 Egg Detection: Traditional Computer Vision vs Deep Learning

A comparative study implementing egg detection using traditional computer vision techniques (OpenCV) and deep learning (Faster R-CNN).

> **📝 Note:** This project was developed in 2020 as part of the STEP Program at Tokyo University of Agriculture and Technology. The code may not run on modern systems due to outdated dependencies, but is preserved here to showcase the methodology and findings.

![Python](https://img.shields.io/badge/Python-3.6.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1.x-orange)
![Status](https://img.shields.io/badge/Status-Archive-lightgrey)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Methods](#methods)
- [Project Structure](#project-structure)
- [Results](#results)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)
- [References](#references)

---

## Overview

Object detection is an essential computer vision technique that enables classification and localization of objects in images. This pilot study implements and compares two approaches for egg detection:

1. **Traditional Computer Vision** - Using handcrafted features with OpenCV
2. **Deep Learning** - Using Faster R-CNN (Region-based Convolutional Neural Network)

The goal was to understand the strengths and limitations of each approach for potential applications in automated egg collection systems in the egg industry.

---

## Key Findings

| Metric | Traditional CV | Faster R-CNN |
|--------|---------------|--------------|
| **Precision** | 99% | 99% |
| **Data Required** | Minimal | Large (1,066 images) |
| **Training Time** | None | ~200,000 iterations |
| **Feature Learning** | Manual/Handcrafted | Automatic/Semantic |

### Summary

- Both methods achieved **99% precision** in egg detection
- Traditional CV works well for simplified, controlled environments with minimal data
- Faster R-CNN can learn deeper, semantic features and adapt to various settings and environments
- Deep learning requires significantly more data and computational resources

---

## Methods

### Traditional Computer Vision Approach

The traditional pipeline uses handcrafted feature extraction:

```
Input Image → BGR-HSV Conversion → Channel Split → Morphological Closing 
→ Distance Transform → Template Creation → Template Matching 
→ Local Maxima Detection → Contour Finding → Output
```

**Tools Used:**
- Python 3.6.10
- OpenCV library
- Jupyter Notebook
- PyQt5 (GUI)

### Deep Learning Approach (Faster R-CNN)

The Faster R-CNN architecture consists of:
- **Region Proposal Network (RPN)** - Generates candidate object regions
- **RoI Pooling** - Extracts fixed-size features from proposals
- **Classifier** - Predicts object class and refines bounding boxes

**Implementation Steps:**
1. Dataset preparation (191 images → 1,066 via data augmentation)
2. Image labeling using LabelImg (Pascal VOC format)
3. TFRecord generation for TensorFlow
4. Transfer learning from `faster_rcnn_inception_v2_coco_2018_01_28`
5. Training for 200,000 iterations
6. Model export as inference graph (.pb files)

**Tools Used:**
- Python 3.6.10
- TensorFlow Object Detection API
- LabelImg for annotation
- TensorBoard for visualization

---

## Project Structure

```
egg-detection-cv-vs-dl/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies (archived)
│
├── docs/
│   └── Final_Report.pdf         # Complete research report
│
├── traditional_cv/
│   ├── egg_detection.py         # Main detection script
│   ├── gui_app.py               # PyQt5 GUI application
│   └── utils/                   # Helper functions
│
├── faster_rcnn/
│   ├── config/                  # Training configuration files
│   ├── data/
│   │   ├── train/               # Training images
│   │   ├── test/                # Test images
│   │   └── annotations/         # XML annotation files
│   ├── models/                  # Trained model files (.pb)
│   ├── scripts/
│   │   ├── generate_tfrecord.py # TFRecord generation
│   │   ├── train.py             # Training script
│   │   └── detect.py            # Detection script
│   └── label_map.pbtxt          # Class label mapping
│
└── images/
    ├── results/                 # Detection result screenshots
    ├── architecture/            # Model architecture diagrams
    └── samples/                 # Sample egg images
```

---

## Results

### Traditional Computer Vision
Successfully detected and counted 130 eggs with 99% precision on conveyor belt video footage.

### Faster R-CNN
Achieved 99% precision at global step 5,853. The loss function showed consistent decrease over 200,000 training iterations, indicating stable learning.

### Comparison Table

| Aspect | Faster R-CNN | Traditional CV |
|--------|--------------|----------------|
| Learning Type | End-to-end | Handcrafted features |
| Feature Learning | Semantic, high-level | Limited, manual |
| Data Requirements | Large datasets | Minimal |
| Training Time | Hours/Days | None |
| Hardware Needs | GPU recommended | CPU sufficient |
| Adaptability | High (various environments) | Low (specific tasks) |
| Engineering Skill | ML/DL knowledge | CV expertise |

---

## Dependencies

> ⚠️ **Warning:** These are the original 2020 dependencies. They may have compatibility issues with modern systems.

```
Python==3.6.10
tensorflow==1.x
opencv-python==4.x
numpy
pandas
Pillow
matplotlib
lxml
PyQt5
```

### Hardware Used
- MacBook Pro 13"
- Processor: 2.5 GHz Dual-Core Intel Core i7

---

## ⚠️ Compatibility Notice

This project was developed in **July 2020** and uses:
- Python 3.6.10 (EOL: December 2021)
- TensorFlow 1.x (superseded by TensorFlow 2.x)
- Older OpenCV versions

**The code is preserved for educational and portfolio purposes.** If you wish to run similar experiments today, consider:
- Upgrading to Python 3.9+
- Using TensorFlow 2.x or PyTorch
- Using newer object detection models (YOLOv8, DETR, etc.)

---

## Acknowledgments

This project was completed as part of the **STEP (Science and Technology Exchange Program)** at **Tokyo University of Agriculture and Technology** (2019-2020).

**Supervisor:** Prof. Seiji Hotta, Associate Professor

Special thanks to:
- The STEP program for the opportunity
- Tokyo University of Agriculture and Technology
- The open-source community for tools and resources

---

## References

1. Wu, X., Sahoo, D., & Hoi, S. C. H. (2020). Recent advances in deep learning for object detection. *Neurocomputing*, 396, 39-64.

2. Li, G., Xu, Y., Zhao, Y., Du, Q., & Huang, Y. (2020). Evaluating convolutional neural networks for cage-free floor egg detection. *Sensors*, 20(2), 1-17.

3. Ren, S., He, K., Girshick, R., & Sun, J. (2017). Faster R-CNN: Towards real-time object detection with region proposal networks. *IEEE TPAMI*, 39(6), 1137-1149.

4. Ursul, I. "How we wrote chicken egg counter on a Raspberry PI." [Link](https://ivanursul.com/counting-eggs-in-opencv)

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👤 Author

**Oulaiphone Ouankhamchanh**

- GitHub: [@Oulaiphone](https://github.com/Oulaiphone)

---

*Last Updated: 2020 | Archived Project*
