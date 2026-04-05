# Flood Detection for ANRF AISE Hackathon

This project is my submission for the ANRF AISE Hackathon, focusing on flood detection using semantic segmentation. The primary objective is to accurately classify pixels in satellite imagery into three categories: No Flood, Flood, and Water Body.

## Methodology

The project evolved through several stages of experimentation with different deep learning models, backbones, and techniques.

### Initial Approach
I began with an **ADLinkNet** architecture using an **Xception** backbone. While this provided a solid baseline, I explored lighter models to improve performance and training efficiency.

### Model & Backbone Experimentation
My experimentation included various combinations:
- **ADLinkNet** with **ResNet18**, which showed promising results.
- **UNet++** with **ResNet34** and **EfficientNet-B3/B5** backbones to leverage more complex feature fusion with nested and dense skip connections.
- A return to **ADLinkNet** with **ResNet18** and **EfficientNet-B5**.

### Loss Functions and Optimization
To tackle the class imbalance and focus on the critical "Flood" class, I implemented and tuned a combination of loss functions:
- **Focal Loss** and **Dice Loss** for handling class imbalance.
- **Tversky Loss** to more heavily penalize false negatives (missed flood pixels).
- **Boundary Loss**, introduced later in the training, to improve segmentation along the edges of water bodies.

### Feature Engineering
To better distinguish between floodwater and permanent water bodies, I engineered additional input channels beyond the standard RGB and SAR bands, including:
- **MNDWI (Modified Normalized Difference Water Index)**
- **SAR Difference (HH - HV)**

This journey of iterative development and experimentation is documented in [thoughts-and-prompts.md](thoughts-and-prompts.md).

## Getting Started

### Download
Clone the repository to your local machine:
```bash
git clone <repository-url>
cd anrf-aisehack-edition-1
```

# Setup

uv is a fast Python package manager. Install it first if you haven't:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a virtual environment:

```bash
# Create virtual environment
uv venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

Install PyTorch with CUDA support (adjust CUDA version as needed):

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install remaining dependencies:

```bash
uv pip install -r requirements.txt
```

Alternative:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source venv/bin/activate

# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

# Inference
To run inference on the prediction dataset, execute the inference.py script from the root directory of the project:
```bash
uv run inference.py
# OR
python models/inference.py
```

```bash

## 📄 requirements.txt (copyable)

```txt
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
pytorch-lightning>=2.0.0
numpy>=1.24.0
rasterio>=1.3.0
albumentations>=1.3.0
segmentation-models-pytorch>=0.3.0
pandas>=2.0.0
scipy>=1.10.0
kagglehub>=0.1.0
timm>=0.9.0
torchmetrics>=1.0.0
scikit-learn>=1.3.0
```

# Quick Setup
For a complete one-liner after cloning:
```bash
# Linux
uv venv && source .venv/bin/activate && uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && uv pip install -r requirements.txt

# Windows
uv venv; .venv\Scripts\activate; uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; uv pip install -r requirements.txt
```

# Resources
A list of courses, papers, and videos that I referred to during this hackathon is available in resources-reffered.md.

# License
This project is licensed under the ANRF Open License. See the LICENSE file for details.