# AI Tools Used
- Claude (Sonnet 4.6) (Web Interface)
- Gemini (Fast/Pro Mode) (Web Interface)

# Prompts
Prompt1:
To Claude (Sonnet 4.6)-
```
Provide me a code implementing ADLinkNet+Xception as Backbone for flood detection
In Phase 2, the labeling scheme is extended from binary to three classes:

0 – No Flood
1 – Flood
2 – Water Body

Evaluation
Ground-truth labels for a subset of test patches will be withheld, and participants will be required to predict one of the three classes (no flood, flood, water body) for each pixel.

Model performance will be evaluated using:

Per-class IoU, with particular emphasis on the flood class
Mean Intersection over Union (mIoU) across all three classes
Overall pixel-wise accuracy
This evaluation setup ensures that models are not only accurate overall but also effective in distinguishing floodwater from existing water bodies, one of the key challenges introduced in Phase 2.

Submission File
For each image ID in the test set, you must submit a run-length encoded (RLE) flood mask representing predicted flood pixels.

The submission file must be a CSV with a header and follow this format:

id,rle_mask
20240529_EO4_RES2_fl_pid_080,1 12 45 8
20240529_EO4_RES2_fl_pid_081, 300 20
20240529_EO4_RES2_fl_pid_082,120 15 200 10
...
Column Description:

id : Unique identifier for each test image. Derived from the TIFF filename by removing the suffix _image.tif.

rle_mask : Run-Length Encoding (RLE) of predicted flood pixels. Encoding follows column-major order (top-to-bottom, then left-to-right) as per Kaggle convention, with pixel indexing starting at 1. If no flood pixels are predicted, this field should be left empty.

Important:

The submission must include all image IDs from the test set.
The rle_mask field must not contain any empty or null values.
If no pixels are predicted for the flood class (i.e., the mask is empty), it must be represented as: 0 0
and not left blank.

# Attachment:
# Phase 1 Notebook
```

Basic Debugging:
To Gemini (Fast/Pro Mode)

Prompt2:
```
Provide me backbone of ResNet18 instead of Xception, provide me code cell by cell where modification is necessary
# Attachment:
# Phase 2 Notebook 1
```

Basic Debugging:
To Gemini (Fast/Pro Mode)

Prompt3:
To Claude (Sonnet 4.6)-
```
1. Architecture: Move to Unet++ or MAET
While AD-LinkNet is efficient, it often lacks the multi-scale feature fusion necessary for thin flood boundaries.
* Recommendation: Transition to Unet++ with a ResNet34 or EfficientNet-B3 backbone. * Why: Unet++ uses nested, dense skip connections that bridge the semantic gap between encoder and decoder features better than the standard LinkNet structure. EfficientNet-B3 provides a more sophisticated feature extraction capability than ResNet18 without the massive parameter count of Xception.
* Backbone Tweak: Continue using your 8-channel input logic but initialize the first layer weights by averaging the ImageNet weights across all 8 channels (instead of just the first 3) to give the SAR and NIR bands a better starting point.
2. Refining the "Tri-Loss" Strategy
Your current loss weights (0.5 for background, 10.0 for flood, 3.0 for water) are aggressive, but the Boundary Loss might be fighting the Focal Loss in early stages.
* Annealed Boundary Loss: Don't start with Boundary Loss at $0.2$ weight. Start at $0$ and linearly increase its weight to $0.3$ after Epoch 15. This allows the model to learn the "blobs" (Dice/Focal) before obsessing over the "edges" (Boundary).
* Class Weighting: Since you are evaluated specifically on Flood IoU, try a Tversky Loss instead of Dice.
   * Tversky Loss ($\alpha=0.3, \beta=0.7$) penalizes False Negatives (missed flood pixels) more heavily than False Positives.
* Formula:
$$T(P, G) = \frac{|P \cap G|}{|P \cap G| + \alpha|P \setminus G| + \beta|G \setminus P|}$$

4. Feature Engineering: Beyond NDWI
You have SAR HH, HV, and NIR. Add the following to your 8-channel stack to separate permanent water from flood:
* MNDWI: $(Green - SWIR) / (Green + SWIR + \epsilon)$. It's better at suppressing "noise" from built-up areas compared to NDWI.
* SAR Difference: $(HH - HV)$. This often highlights the surface roughness change between turbulent flood water and calm permanent water.

5. Post-Processing: The "Water Mask" Constraint
Since permanent water (Class 2) is often geographically static, you can use a Permanent Water Mask (if available or derivable from the NIR band) to "gate" your predictions.
* If your model predicts "Flood" (Class 1) on a pixel that has a high NIR-based water probability and exists in every historical frame, it's likely "Water" (Class 2).
* Use a Conditional Random Field (CRF) as a final step to smooth out the RLE masks and remove isolated single-pixel "salt and pepper" noise.

So I want to try ResNet34 and EfficientNet-B3 with UNET++

Also make sure you reason the above application, apply only if better.

Provide me the modified code for notebook to add the ResNet34, EfficientNet-B3 with UNET++ and feature engineering, refining loss strategy.

# Attachment:
# Phase 2 Notebook 2
```

Basic Debugging:
To Gemini (Fast/Pro Mode)

# My Chain of Thoughts:
Based on my reference to Comparison of Backbones for Semantic
Segmentation Network: https://iopscience.iop.org/article/10.1088/1742-6596/1544/1/012196/pdf, and previous hands-on of CNN Architectures like ResNet, EfficientNet during kaggle assignments from my course BSDA2001, I tried ADLinkNet+Xception, but later trying with different loss weights, epochs and augmentation it almost plateaued, hence I tried smaller model ResNet18 with ADLinkNet, after it performed well, I thought of trying other similar models and hence chose ResNet34 and EfficientNet-B3.