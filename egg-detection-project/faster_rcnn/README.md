# Faster R-CNN - Deep Learning Egg Detection

This folder contains the deep learning implementation for egg detection using Faster R-CNN with TensorFlow Object Detection API.

## Architecture

Faster R-CNN is a two-stage object detector consisting of:

1. **Backbone Network** - Inception V2 (pre-trained on COCO dataset)
2. **Region Proposal Network (RPN)** - Generates candidate object regions
3. **RoI Pooling Layer** - Extracts fixed-size features from proposals
4. **Classification Head** - Predicts class and refines bounding boxes

## Folder Structure

```
faster_rcnn/
├── README.md                              # This file
├── label_map.pbtxt                        # Class label mapping (egg: 1)
│
├── config/
│   └── faster_rcnn_inception_v2_coco.config  # Training configuration
│
└── scripts/
    ├── generate_tfrecord.py       # Convert annotations to TFRecords
    ├── xml_to_csv.py              # Convert XML annotations to CSV
    ├── model_main.py              # Training script
    └── detection_inference.py     # Run detection on new images

Note: The trained model file (frozen_inference_graph.pb) is not included 
due to its large size (57MB). Users can retrain the model using the 
provided scripts and configuration.
```

## Dataset

| Split | Images | Notes |
|-------|--------|-------|
| Original | 191 | Various camera angles |
| Augmented | 1,066 | After data augmentation |
| Training | 1,013 | 95% of augmented data |
| Testing | 53 | 5% of augmented data |

### Data Augmentation Techniques Used
- Horizontal/Vertical flips
- Rotation
- Brightness adjustment
- Scaling

## Training Configuration

- **Base Model:** `faster_rcnn_inception_v2_coco_2018_01_28`
- **Iterations:** 200,000
- **Batch Size:** 1 (limited by hardware)
- **Learning Rate:** Default from config

## Label Map

```protobuf
item {
    id: 1
    name: 'egg'
}
```

## Usage (Original - 2020)

### Training
```bash
# Note: Requires TensorFlow 1.x Object Detection API setup

python train.py \
    --pipeline_config_path=config/faster_rcnn.config \
    --model_dir=models/
```

### Detection
```python
from detect import EggDetectorRCNN

detector = EggDetectorRCNN('models/inference_graph/frozen_inference_graph.pb')
detections = detector.detect('path/to/image.jpg')
```

### Monitoring Training
```bash
tensorboard --logdir=models/
```

## Results

- **Precision:** 99% (at global step 5,853)
- **Final Loss:** ~0.45 (at 200,000 iterations)
- **Processing Speed:** Suitable for batch processing

## Hardware Requirements (2020)

- **Used:** MacBook Pro 13", 2.5 GHz Dual-Core Intel Core i7
- **Recommended:** GPU with CUDA support for faster training
- **Training Time:** Several hours to days depending on hardware

## Limitations

- Requires large annotated dataset
- Long training time without GPU
- TensorFlow 1.x is now deprecated
- Model size is large (~100MB+)

## Modern Alternatives

If recreating this project today, consider:
- **YOLOv8** - Faster, easier to use
- **DETR** - Transformer-based detection
- **TensorFlow 2.x** - Updated API
- **PyTorch + Detectron2** - Facebook's detection library

---

## References

- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497) - Ren et al., 2015
- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [LabelImg Tool](https://github.com/tzutalin/labelImg)

---

*Part of the Egg Detection comparative study - STEP Program 2020*
