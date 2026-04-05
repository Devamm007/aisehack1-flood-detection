import time
start = time.time()

import numpy as np
import pandas as pd
import gc
import random
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import albumentations as A
from pathlib import Path
from itertools import filterfalse as ifilterfalse
import cv2
from scipy import ndimage
import kagglehub
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path('../data')
IMG_DIR      = DATA_ROOT / 'image'
LABEL_DIR    = DATA_ROOT / 'label'
SPLIT_DIR    = DATA_ROOT / 'split'
PRED_IMG_DIR = DATA_ROOT / 'prediction' / 'image'

# ─── Hyperparameters ──────────────────────────────────────────────────────────
NUM_CLASSES     = 3
IN_CHANNELS     = 15                # Increased for AWEI inclusion
PATCH_SIZE      = 512
BATCH_SIZE      = 2                 
ACCUM_GRAD      = 4                 
NUM_EPOCHS      = 100               # SWA needs a longer tail to converge
LR              = 2e-4              
WEIGHT_DECAY    = 1e-3              # Increased to fight overfitting on 69 images
N_FOLDS         = 5
SEED            = 42
ENCODER         = 'efficientnet-b5' 
ENCODER_WEIGHTS = 'imagenet'

# OPTIMIZATION
USE_SWA         = True              
FLOOD_THRESHOLD = 0.38

# Dropout rate for decoder
DECODER_DROPOUT = 0.3

# Loss weights
TVERSKY_ALPHA   = 0.3      # FP penalty
TVERSKY_BETA    = 0.7      # FN penalty (flood recall)
LOVASZ_WEIGHT   = 0.5
TVERSKY_WEIGHT  = 0.5

# Class weights (estimated from training set: background ~80%, flood ~15%, water ~5%)
CLASS_WEIGHTS   = [0.2, 3.0, 1.5]

pl.seed_everything(SEED)
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print(f"Encoder: {ENCODER}, Params: ~12M")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()

def load_ids(name):
    with open(SPLIT_DIR / f'{name}.txt') as f:
        return [l.strip() for l in f if l.strip()]

train_ids = load_ids('train')
val_ids   = load_ids('val')
test_ids  = load_ids('test')
pred_ids  = load_ids('pred')

ALL_TRAIN_IDS = train_ids + val_ids   # 69 images for 5-fold CV
PRED_IDS      = pred_ids

print(f"Train+Val (for CV): {len(ALL_TRAIN_IDS)}")
print(f"Test (held-out val): {len(test_ids)}")
print(f"Pred (final submission): {len(PRED_IDS)}")

B_HH, B_HV, B_GREEN, B_RED, B_NIR, B_SWIR = 0, 1, 2, 3, 4, 5

# Per-band normalisation stats (computed from training set)
RAW_MEANS = np.array([841.13, 371.62, 1734.18, 1588.31, 1742.85, 1218.56], dtype=np.float32)
RAW_STDS  = np.array([473.71, 170.36,  623.05,  612.85,  564.58,  528.09], dtype=np.float32)


def load_raw_image(img_path):
    """Load 6-band GeoTIFF → float32 [6, H, W], NaN/Inf replaced with 0."""
    with rasterio.open(img_path) as src:
        raw = src.read().astype(np.float32)
    return np.where(np.isfinite(raw), raw, 0.0)


def engineer_15ch(raw):
    """Build 15-channel tensor. 
    Adds AWEI for shadow suppression and a SAR Mean Intensity band.
    """
    eps = 1e-6
    hh, hv       = raw[B_HH], raw[B_HV]
    green, red   = raw[B_GREEN], raw[B_RED]
    nir, swir    = raw[B_NIR], raw[B_SWIR]

    # Z-score normalization
    raw_norm = (raw - RAW_MEANS[:, None, None]) / (RAW_STDS[:, None, None] + eps)

    # Spectral Indices
    mndwi = np.clip((green - swir) / (green + swir + eps), -1.0, 1.0)
    ndvi  = np.clip((nir - red) / (nir + red + eps), -1.0, 1.0)
    
    # AWEI (sh) - Excellent for urban/mountainous shadow areas
    awei = 4.0 * (green - swir) - (0.25 * nir + 2.75 * red)
    awei = np.clip(awei / 5000.0, -1.0, 1.0) 

    # SAR Features
    hh_log = np.log1p(np.clip(hh, 0, None))
    hv_log = np.log1p(np.clip(hv, 0, None))
    sar_ratio = np.log(np.abs(hh) + eps) - np.log(np.abs(hv) + eps)

    stack = np.stack([
        raw_norm[B_HH], raw_norm[B_HV], 
        raw_norm[B_GREEN], raw_norm[B_RED],
        raw_norm[B_NIR], raw_norm[B_SWIR],
        mndwi, np.tanh(3.0 * mndwi),     
        ndvi, awei,                      # Channel 9: AWEI
        sar_ratio, hh_log - hv_log,      
        -hv_log, -mndwi,                 # Water proxies
        (hh_log + hv_log) / 2.0          # Channel 14: SAR Mean
    ], axis=0).astype(np.float32)

    return np.clip(stack, -10.0, 10.0)

# Sanity check
_t = engineer_15ch(np.random.rand(6, 512, 512).astype(np.float32) * 1000)
print(f"Feature shape: {_t.shape}  |  Range: [{_t.min():.2f}, {_t.max():.2f}]")

del _t
gc.collect()

TRAIN_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # instead apply affine
    A.Affine(rotate=(-15, 15), translate_percent=0.1, scale=(0.85, 1.15), shear=(-10, 10), mode=0, p=0.4),
    A.OneOf([
        A.GaussNoise(var_limit=(0.001, 0.01), p=1.0),
        A.GaussianBlur(blur_limit=3, p=1.0),
    ], p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
])

VAL_AUG = None   # No augmentation at test time (TTA handles it separately)

def mixup_cutmix_collate(batch, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    images, masks = default_collate(batch)  # images: [B,C,H,W], masks: [B,H,W]
    if random.random() > prob:
        return images, masks

    use_cutmix = random.random() > 0.5
    lam = np.random.beta(mixup_alpha, mixup_alpha) if not use_cutmix else np.random.beta(cutmix_alpha, cutmix_alpha)

    batch_size = images.size(0)
    index = torch.randperm(batch_size)

    if use_cutmix:
        H, W = images.shape[2], images.shape[3]
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)
        bw = int(np.round(W * np.sqrt(1 - lam)))
        bh = int(np.round(H * np.sqrt(1 - lam)))
        x1 = max(0, cx - bw // 2)
        x2 = min(W, cx + bw // 2)
        y1 = max(0, cy - bh // 2)
        y2 = min(H, cy + bh // 2)
        # Apply to images
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        # Apply to masks - note the batch dimension
        masks[:, y1:y2, x1:x2] = masks[index, y1:y2, x1:x2]
        # Adjust lam to actual area
        lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    else:
        # MixUp
        images = lam * images + (1 - lam) * images[index]
        masks = lam * masks + (1 - lam) * masks[index]
        masks = masks.long()
    return images, masks

print("Augmentation pipelines ready.")

class FloodDataset(Dataset):
    """
    Loads 6-band GeoTIFFs, engineers 14-channel tensors, returns 3-class masks.
    Label convention:  0=No Flood  1=Flood  2=Water Body  -1=Ignore
    """
    def __init__(self, file_ids, img_dir, label_dir=None, transform=None):
        self.file_ids  = list(file_ids)
        self.img_dir   = Path(img_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.transform = transform

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]

        # Image
        img_paths = list(self.img_dir.glob(f'*{fid}_image.tif'))
        assert img_paths, f"Image not found for {fid}"
        raw  = load_raw_image(img_paths[0])     # [6, H, W]
        feat = engineer_15ch(raw)               # [14, H, W]
        img_hw = feat.transpose(1, 2, 0)        # [H, W, 14] for albumentations

        # Label
        if self.label_dir is not None:
            lbl_paths = list(self.label_dir.glob(f'*{fid}_label.tif'))
            assert lbl_paths, f"Label not found for {fid}"
            with rasterio.open(lbl_paths[0]) as src:
                mask = src.read(1).astype(np.int64)
            mask = np.where((mask >= 0) & (mask <= 2), mask, -1)
        else:
            mask = np.zeros(feat.shape[1:], dtype=np.int64)

        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=img_hw, mask=mask.astype(np.int32))
            img_hw = aug['image']
            mask   = aug['mask'].astype(np.int64)

        img_t  = torch.from_numpy(img_hw.transpose(2, 0, 1)).float()  # [14,H,W]
        mask_t = torch.from_numpy(mask).long()                         # [H,W]
        return img_t, mask_t


class PredDataset(Dataset):
    """Dataset for final prediction images (no labels)."""
    def __init__(self, file_ids, img_dir):
        self.file_ids = list(file_ids)
        self.img_dir  = Path(img_dir)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fid = self.file_ids[idx]
        img_paths = list(self.img_dir.glob(f'*{fid}_image.tif'))
        assert img_paths, f"Pred image not found for {fid}"
        raw  = load_raw_image(img_paths[0])
        feat = engineer_15ch(raw)
        return torch.from_numpy(feat).float(), fid

print("Dataset classes defined.")

def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors"""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss"""
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_softmax_flat(probas, labels, classes='present'):
    """Multi-class Lovasz-Softmax loss"""
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        errors = (probas[:, c] - fg).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean()

def lovasz_softmax(logits, labels, ignore_index=-1, per_image=True):
    """Multi-class Lovasz-Softmax loss for semantic segmentation"""
    logits = logits.permute(0, 2, 3, 1).contiguous()
    labels = labels.contiguous()
    if per_image:
        def treat_image(log_i, lab_i):
            log_i = log_i.view(-1, log_i.size(-1))
            lab_i = lab_i.view(-1)
            mask = lab_i != ignore_index
            log_i = log_i[mask]
            lab_i = lab_i[mask]
            probas = F.softmax(log_i, dim=1)
            return lovasz_softmax_flat(probas, lab_i)
        losses = [treat_image(log_i, lab_i) for log_i, lab_i in zip(logits, labels)]
        loss = torch.stack(losses).mean()
    else:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        mask = labels != ignore_index
        logits = logits[mask]
        labels = labels[mask]
        probas = F.softmax(logits, dim=1)
        loss = lovasz_softmax_flat(probas, labels)
    return loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, ignore_index=-1, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.class_weights = torch.tensor(class_weights) if class_weights else None

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index)
        tgt = targets.clone()
        tgt[~valid] = 0
        tgt_onehot = F.one_hot(tgt, num_classes=logits.size(1)).permute(0,3,1,2).float()
        probs = probs * valid.unsqueeze(1).float()

        loss = 0.0
        for c in range(logits.size(1)):
            p = probs[:, c]
            g = tgt_onehot[:, c]
            tp = (p * g).sum()
            fp = (p * (1 - g)).sum()
            fn = ((1 - p) * g).sum()
            tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
            class_weight = self.class_weights[c] if self.class_weights is not None else 1.0
            loss += class_weight * (1 - tversky)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-1, lovasz_weight=0.5, tversky_weight=0.5,
                 tversky_alpha=0.3, tversky_beta=0.7, class_weights=None):
        super().__init__()
        self.lovasz_weight = lovasz_weight
        self.tversky_weight = tversky_weight
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta,
                                   ignore_index=ignore_index, class_weights=class_weights)

    def forward(self, logits, targets):
        # Use local lovasz_softmax function
        lovasz = lovasz_softmax(logits, targets, ignore_index=-1, per_image=True)
        tversky = self.tversky(logits, targets)
        return self.lovasz_weight * lovasz + self.tversky_weight * tversky

class FloodSegModel(pl.LightningModule):
    """
    UNet++ with EfficientNet-B5 encoder and scSE attention.
    14-channel input, 3-class output.
    Differential learning rates: encoder LR/10, decoder LR.
    """
    def __init__(self, in_channels=14, num_classes=3,
                 encoder_name='efficientnet-b3', encoder_weights='imagenet',
                 decoder_dropout=0.3, lr=3e-4, weight_decay=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_attention_type='scse',
            decoder_use_batchnorm=True,
            decoder_dropout=decoder_dropout,   # ← add dropout
        )
        self.criterion = CombinedLoss(
            num_classes=num_classes,
            ignore_index=-1,
            lovasz_weight=0.5,
            tversky_weight=0.5,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            class_weights=CLASS_WEIGHTS
        )
        self._reset_iou('train')
        self._reset_iou('val')

    def _reset_iou(self, split):
        for attr in ['tp', 'fp', 'fn']:
            setattr(self, f'{split}_{attr}', torch.zeros(self.num_classes))

    def _update_iou(self, logits, masks, split):
        preds = torch.argmax(logits, dim=1)
        valid = masks != -1
        for c in range(self.num_classes):
            pc = (preds == c) & valid
            gc = (masks == c) & valid
            getattr(self, f'{split}_tp')[c] += (pc & gc).sum().float().cpu()
            getattr(self, f'{split}_fp')[c] += (pc & ~gc).sum().float().cpu()
            getattr(self, f'{split}_fn')[c] += (~pc & gc).sum().float().cpu()

    def _log_epoch_iou(self, split):
        tp   = getattr(self, f'{split}_tp')
        fp   = getattr(self, f'{split}_fp')
        fn   = getattr(self, f'{split}_fn')
        iou  = tp / (tp + fp + fn + 1e-6)
        miou = iou.mean()
        self.log(f'{split}_mIoU',      miou,   prog_bar=True,  sync_dist=True)
        self.log(f'{split}_IoU_BG',    iou[0], prog_bar=False, sync_dist=True)
        self.log(f'{split}_IoU_Flood', iou[1], prog_bar=True,  sync_dist=True)
        self.log(f'{split}_IoU_Water', iou[2], prog_bar=True,  sync_dist=True)
        self._reset_iou(split)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        self._update_iou(logits.detach(), masks, 'train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self._log_epoch_iou('train')

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        loss   = self.criterion(logits, masks)
        self._update_iou(logits.detach(), masks, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self._log_epoch_iou('val')

    def configure_optimizers(self):
        enc_params = list(self.model.encoder.parameters())
        dec_params = (list(self.model.decoder.parameters()) +
                      list(self.model.segmentation_head.parameters()))
        optimizer = torch.optim.AdamW([
            {'params': enc_params, 'lr': self.lr * 0.1},
            {'params': dec_params, 'lr': self.lr},
        ], weight_decay=self.hparams.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

print("FloodSegModel defined.")

def tta_predict(model, images):
    aug_probs = []
    with torch.no_grad():
        # Original
        aug_probs.append(torch.softmax(model(images), dim=1))
        # Horizontal flip
        aug_probs.append(torch.softmax(model(torch.flip(images, [3])), dim=1).flip([3]))
        # Vertical flip
        aug_probs.append(torch.softmax(model(torch.flip(images, [2])), dim=1).flip([2]))
        # Rot90
        r90 = torch.rot90(images, 1, [2,3])
        aug_probs.append(torch.softmax(model(r90), dim=1).rot90(-1, [2,3]))
        # Rot270
        r270 = torch.rot90(images, 3, [2,3])
        aug_probs.append(torch.softmax(model(r270), dim=1).rot90(-3, [2,3]))
                
    return torch.stack(aug_probs).mean(0)

def postprocess_flood_mask(mask: np.ndarray, min_area=10, fill_holes=True) -> np.ndarray:
    """
    mask: binary flood mask (0/1)
    Remove small connected components and optionally fill holes.
    """
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(num_features+1))
    # Keep only components larger than min_area
    for comp in range(1, num_features+1):
        if sizes[comp] < min_area:
            mask[labeled == comp] = 0
    # Fill small holes (optional)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
    return mask

def run_ensemble_inference(ckpt_list, pred_ids, pred_img_dir, batch_size=2):
    """
    ckpt_list: [(path, val_score), ...]
    Returns: [(fid, pred_mask_np), ...] where pred_mask_np is [H,W] with values 0/1/2
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_ds = PredDataset(pred_ids, pred_img_dir)
    pred_dl = DataLoader(pred_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Softmax ensemble weights from val scores
    scores  = np.array([s for _, s in ckpt_list], dtype=np.float32)
    weights = np.exp(scores) / np.exp(scores).sum()
    print(f"Ensemble: {len(ckpt_list)} models | weights: {[f'{w:.3f}' for w in weights]}")

    # Accumulate weighted probability maps (one entry per sample)
    cum_probs = None   # list of [C, H, W] tensors

    for model_i, ((ckpt_path, _), w) in enumerate(zip(ckpt_list, weights)):
        print(f"  [{model_i+1}/{len(ckpt_list)}] {Path(ckpt_path).name}")
        m = FloodSegModel.load_from_checkpoint(ckpt_path).to(device)
        m.eval()

        model_probs = []
        for imgs, _ in pred_dl:
            imgs = imgs.to(device)
            p    = tta_predict(m, imgs)   # [B, C, H, W]
            model_probs.extend(p.cpu().unbind(0))

        if cum_probs is None:
            cum_probs = [w * p for p in model_probs]
        else:
            for j, p in enumerate(model_probs):
                cum_probs[j] += w * p

        del m, model_probs
        gc.collect(); torch.cuda.empty_cache()
        print(f"    done.")

    # Collect fids in order
    fid_list = []
    for _, fids in DataLoader(pred_ds, batch_size=1, shuffle=False):
        fid_list.extend(fids)

    results = []
    for fid, prob_map in zip(fid_list, cum_probs):
        # prob_map: [3, H, W]
        # Instead of argmax, we manually define the classes
        
        # 1. Start with Background (Class 0)
        final_mask = np.zeros((512, 512), dtype=np.uint8)
        
        # 2. Assign Water Body (Class 2) if prob is high
        water_mask = (prob_map[2] > 0.5).numpy()
        final_mask[water_mask] = 2
        
        # 3. Assign Flood (Class 1) using the OPTIMIZED threshold
        # This usually provides a massive IoU boost on the LB
        flood_mask = (prob_map[1] > FLOOD_THRESHOLD).numpy()
        final_mask[flood_mask] = 1
        
        # Post-process the flood mask specifically
        cleaned_flood = (final_mask == 1).astype(np.uint8)
        cleaned_flood = postprocess_flood_mask(cleaned_flood, min_area=120)
        
        # Re-apply cleaned flood to final mask
        final_mask[final_mask == 1] = 0
        final_mask[cleaned_flood == 1] = 1
        
        results.append((fid, final_mask))

    return results

def mask_to_rle(mask: np.ndarray) -> str:
    """
    Binary flood mask → RLE string (column-major, 1-indexed).
    Returns '0 0' if no flood pixels (as per competition rules).
    """
    pixels = mask.flatten(order='F')          # column-major (Kaggle convention)
    pixels = np.concatenate([[0], pixels, [0]])
    runs   = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle_str = ' '.join(str(x) for x in runs)
    return rle_str if rle_str else '0 0'

model_dir = Path('aisehack-flood-unetpp-efficientnet-b5-pytorch-v3-v1')
print(f"Using model from: {model_dir}")

# Load metadata
meta_path = model_dir / "metadata.json"
if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
    encoder_name = meta["config"]["encoder"]
    in_channels  = meta["config"]["in_channels"]
    num_classes  = meta["config"]["num_classes"]
    # Get list of checkpoint files from metadata
    ckpt_files = [model_dir / e["path"] for e in meta["ensemble_checkpoints"]]
else:
    # Fallback: use all .ckpt files in directory
    encoder_name = "efficientnet-b5"   # default
    in_channels = 14
    num_classes = 3

ckpt_files = sorted(model_dir.glob("*.ckpt"))

print(f"Found {len(ckpt_files)} checkpoint(s):")
for f in ckpt_files:
    print(f"  {f.name}")

def predict_flood(image_paths, ckpt_paths, batch_size=1, min_area=50):
    """
    image_paths: list of paths to 6‑band GeoTIFFs
    ckpt_paths: list of checkpoint paths (ensemble)
    Returns: list of (image_id, flood_mask_np) where flood_mask is binary uint8
    """
    # Load all models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    for cp in ckpt_paths:
        model = FloodSegModel(encoder_name=encoder_name, in_channels=in_channels, num_classes=num_classes)
        state = torch.load(cp, map_location='cpu')['state_dict']
        state = torch.load(cp, map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(state) 
        
        model.to(device)
        model.eval()
        models.append(model)

    results = []
    for img_path in image_paths:
        raw = load_raw_image(img_path)
        feat = engineer_15ch(raw)           
        
        # 3. Change this from .to('cuda')
        img_tensor = torch.from_numpy(feat).unsqueeze(0).to(device) # [1,14,H,W]

        # Ensemble TTA
        ensemble_prob = None
        for model in models:
            prob = tta_predict(model, img_tensor)   # [1,3,H,W]
            if ensemble_prob is None:
                ensemble_prob = prob
            else:
                ensemble_prob += prob
        ensemble_prob /= len(models)

        # Get flood class (class index 1)
        flood_prob = ensemble_prob[0, 1].cpu().numpy() 

        # 2. Apply the manual threshold
        # Using a value between 0.3 and 0.45 often yields better IoU than argmax
        flood_mask = (flood_prob > FLOOD_THRESHOLD).astype(np.uint8)
        
        # pred = torch.argmax(ensemble_prob, dim=1).squeeze(0).cpu().numpy()  # [H,W]
        # flood_mask = (pred == 1).astype(np.uint8)
        
        flood_mask = postprocess_flood_mask(flood_mask, min_area=min_area, fill_holes=True)

        # Extract image ID from filename
        img_id = Path(img_path).stem.replace('_image', '')
        results.append((img_id, flood_mask))

    # Cleanup
    for m in models:
        del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

test_image_folder = PRED_IMG_DIR   # change this
image_paths = list(Path(test_image_folder).glob("*_image.tif"))
print(f"Found {len(image_paths)} images")

# Use all downloaded checkpoints for ensemble
ckpt_paths = [str(p) for p in ckpt_files]

predictions = predict_flood(image_paths, ckpt_paths, batch_size=1, min_area=50)

# Convert to RLE submission
def mask_to_rle(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle_str = ' '.join(str(x) for x in runs)
    return rle_str if rle_str else '0 0'

sub_rows = []
for img_id, flood_mask in predictions:
    sub_rows.append({'id': img_id, 'rle_mask': mask_to_rle(flood_mask)})

df = pd.DataFrame(sub_rows)
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

# ~1min 20s - 1min 25s for 19 images with 5-model ensemble and TTA on a single GPU.
# The post-processing step adds a few seconds but can significantly boost IoU.
print(f"Total inference time: {(time.time() - start)/60:.0f} minutes {(time.time() - start)%60:.0f} seconds for {len(predictions)} images")