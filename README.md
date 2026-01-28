# Fabric Defect Detection Model

This is the official repository for the Fabric Defect Detection Model for TDS Softlink Company. It's main target is to detect defects in fabric in realtime using suitable models (as of now YOLO) through camera.

### Required Tools

- Python 3.11 or above
- IDE or Text Editor

### Step to run

1. clone this repository:

```bash
git clone https://github.com/Muntahi-Safwan/Fabric-Defect.git
```

2. Change Directory:

```bash
cd Fabric-Defect
```

3. Setup Virtual Environment:

```bash
python -m venv env
```

4. Activate Virtual Environment:

```bash
env\Scripts\activate
```

5. Install Dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Configuration & Augmentation Settings

The YOLO model training (`src/training/train_yolo.py`) uses carefully tuned augmentation settings optimized for fabric defect detection. Below is an explanation of each setting:

### Color Augmentations

These simulate real-world lighting and fabric color variations:

| Parameter    | Value | Description                                                           |
| ------------ | ----- | --------------------------------------------------------------------- |
| `hsv_h`      | 0.015 | Hue variation - simulates slight color shifts from different dye lots |
| `hsv_s`      | 0.7   | Saturation variation - important for printed/patterned fabrics        |
| `hsv_v`      | 0.4   | Brightness variation - simulates different lighting conditions        |
| `brightness` | 0.2   | Additional lighting changes during inspection                         |
| `contrast`   | 0.3   | Contrast variations to handle different fabric textures               |
| `saturation` | 0.5   | Extra saturation adjustments for colored fabrics                      |

### Geometric Augmentations

Conservative settings since the camera position is fixed during inspection:

| Parameter     | Value  | Description                                                      |
| ------------- | ------ | ---------------------------------------------------------------- |
| `degrees`     | 0.0    | **No rotation** - camera is fixed, rotation would be unrealistic |
| `translate`   | 0.1    | Small translation (10%) - defects can appear anywhere in frame   |
| `scale`       | 0.5    | Scale variation - handles different defect sizes                 |
| `shear`       | 2.0    | Small shear angle - simulates fabric stretch/tension             |
| `perspective` | 0.0001 | Minimal perspective distortion                                   |

### Flip Augmentations

| Parameter | Value | Description                                                       |
| --------- | ----- | ----------------------------------------------------------------- |
| `fliplr`  | 0.5   | 50% horizontal flip - fabric can run in both directions           |
| `flipud`  | 0.0   | **No vertical flip** - defects have specific orientation patterns |

### Advanced Augmentations

These are critical for improving detection of small fabric defects:

| Parameter    | Value | Description                                                                                 |
| ------------ | ----- | ------------------------------------------------------------------------------------------- |
| `mosaic`     | 1.0   | 100% mosaic augmentation - combines 4 images, excellent for detecting small defects         |
| `mixup`      | 0.2   | 20% mixup - blends two images together for regularization                                   |
| `copy_paste` | 0.3   | 30% copy-paste augmentation - **best for fabric defects**, duplicates defects across images |

### Optimization Settings

| Parameter       | Value  | Description                                       |
| --------------- | ------ | ------------------------------------------------- |
| `lr0`           | 0.01   | Initial learning rate                             |
| `lrf`           | 0.01   | Final learning rate factor (lr0 × lrf = final lr) |
| `momentum`      | 0.937  | SGD momentum for stable training                  |
| `weight_decay`  | 0.0005 | L2 regularization to prevent overfitting          |
| `warmup_epochs` | 3.0    | Gradual warmup for stable training start          |
| `cos_lr`        | True   | Cosine learning rate scheduler for smooth decay   |

### Loss Function Weights

| Parameter | Value | Description                                               |
| --------- | ----- | --------------------------------------------------------- |
| `box`     | 7.5   | Box loss gain - higher weight for accurate bounding boxes |
| `cls`     | 0.5   | Classification loss gain - lower for few defect classes   |
| `dfl`     | 1.5   | Distribution Focal Loss - improves box regression         |

### Training Configuration

| Parameter  | Value | Description                                               |
| ---------- | ----- | --------------------------------------------------------- |
| `epochs`   | 100   | Total training epochs (more for learning fabric patterns) |
| `patience` | 50    | Early stopping if no improvement for 50 epochs            |
| `batch`    | 16    | Batch size per iteration                                  |
| `imgsz`    | 640   | Input image size (640×640 pixels)                         |
| `cache`    | 'ram' | Cache dataset in RAM for faster training                  |
| `amp`      | True  | Mixed precision training for GPU efficiency               |
| `seed`     | 42    | Random seed for reproducibility                           |

---

### Contribution Guideline

No work will be pushed directly into the main branch. For collaborating create a new branch separately termed via `features\{feature_name}`. Make sure to maintain consistency in code. After working on a particular feature create a pull request to the dev branch.
