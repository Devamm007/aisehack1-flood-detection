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

### Setup
It is recommended to use a virtual environment to manage dependencies.

1. **Create and activate a virtual environment:**
   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Inference
To run inference on the prediction dataset, execute the `inference.py` script from the root directory of the project:

```bash
python models/inference.py
```
This will process the images in the `data/prediction/image` directory and generate a `submission.csv` file in the root folder.

## Resources
A list of courses, papers, and videos that I referred to during this hackathon is available in [resources-reffered.md](resources-reffered.md).

## License
This project is licensed under the ANRF Open License. See the [LICENSE](LICENSE) file for details.
