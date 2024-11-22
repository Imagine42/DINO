# DINO-DETR: Deformable DETR with Improvements

## Overview
This project provides a robust object detection system based on **DINO-DETR**, an enhanced version of Deformable DETR.  
It supports text and object detection, bounding box visualization, and is optimized for GPU environments.

---

## Features
- **Input Formats**: Supports images (JPEG, PNG).
- **Text Detection**: Leverages DINO-DETR for accurate text detection with normalized bounding boxes and confidence scores.
- **Visualization**: Bounding boxes, class labels, and confidence scores are rendered on the input image.

---

## Installation

### **Prerequisites**
- Python >= 3.7
- PyTorch >= 1.12
- CUDA-enabled GPU (optional but recommended)

### 1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```


### 2. Install the package

Run the following command to install the package and compile CUDA operators:
```bash
pip install -e .
```
### 3. Verify installation
```bash
python -c "import dino; print('DINO-DETR is installed successfully!')"
```

### 1. Prepare configuration and weights

	•	Download the pre-trained weights from the DINO GitHub page.
	•	Place the weights file in the weights/ directory.

### 2. Example script
```python
from PIL import Image
from dino.detection.text_detection import DinoDetrModel

# Initialize the model
config_path = "dino/configs/DINO/DINO_4scale.py"
weight_path = "dino/weights/DINO_4scale_coco.pth"
dino_detr = DinoDetrModel(config_path, weight_path)

# Load and process an image
image = Image.open("path_to_image.jpg").convert("RGB")
results = dino_detr.detect(image)

# Visualize results
from dino.detection.visualization import visualize_detections
visualize_detections(image, results, threshold=0.5)
```

## Project Structure
```bash
DINO_Text_Detection/
├── dino/                     # Python package
│   ├── configs/              # Model configurations
│   ├── datasets/             # Dataset utilities
│   ├── detection/            # Detection and visualization
│   ├── models/               # Model definitions
│   ├── ops/                  # Custom CUDA operators
│   ├── weights/              # Pre-trained weights (excluded in .gitignore)
│   ├── util/                 # Utility scripts
├── setup.py                  # Installation script
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

## Contributing

Feel free to submit issues or pull requests to improve this repository.

## License

LICNESE
DINO is released under the Apache 2.0 license. Please see the LICENSE file for more information.

Copyright (c) IDEA. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
