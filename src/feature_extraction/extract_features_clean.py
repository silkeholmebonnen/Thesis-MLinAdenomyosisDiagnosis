#!/usr/bin/env python3
"""
Feature Extraction for Single Model
Extracts 2D slice features from a single pretrained model and saves to disk.

"""
import argparse
import os
# Set HuggingFace to offline mode to use cached models
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TORCH_HUB_OFFLINE'] = '1'
# Use local cache directory (check multiple locations for cached models)
#HF_CACHE_LOCAL = '/dcai/users/wesdav/gits/ultrasound_ml/timm/huggingface'
HF_CACHE_ALT = '/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/ultrasound_ml/ml/huggingface'
# Default to local cache
os.environ['HF_HOME'] = HF_CACHE_ALT
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
#print(f"Default HuggingFace cache: {HF_CACHE_LOCAL}")
print(f"Alternate cache available at: {HF_CACHE_ALT}")

import time
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
import tqdm
from mmpretrain.registry import MODELS
from mmdet.registry import TRANSFORMS

# Configuration
MAPPING_FILE = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/images_with_labels_nov25.csv'
NII_PATH = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/kretz/nifti_kretz_converted2'
N_SLICES = 64  # Fixed depth for stacked
BATCH_SIZE = 4096

# Feature output directory
FEATURE_DIR = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/preextracted_features/'

# Label column 
LABEL_COL = 'myometrial cysts' # this is only used to filter for images where we labels for 

###EDIT HERE ##########################################################
#define your models: 'model_type': (model_name, model_provider, weights_path, feat_dim)
# All available models
FE_MODELS = {
   "dinov3": ("vit_base_patch16_dinov3.lvd1689m", "timm", None, 768),
   "usf-mae": ("vit_base_patch16_224.mae", "timm", "/dcai/projects/cu_0014/data/copl/ultrasound_ml/pretrained_weights/USF-MAE_full_pretrain_43dataset_100epochs.pt", 768),
   "ultrasam": ("mmpretrain.ViTSAM", "mmpretrain", "/dcai/projects/cu_0014/data/copl/ultrasound_ml/pretrained_weights/UltraSam.pth", 768)

}

ultrasam_config = {
    "type": "mmpretrain.ViTSAM",
    "arch": "base",
    "img_size": 1024,
    "patch_size": 16,
    "out_channels": 0,
    "use_abs_pos": True,
    "use_rel_pos": True,
    "window_size": 14,
    "out_type": "avg_featmap",
    "init_cfg": {
        "type": "Pretrained",
        "prefix": "backbone.",
        "checkpoint": "/dcai/projects/cu_0014/data/copl/ultrasound_ml/pretrained_weights/UltraSam.pth"
    },
}

def load_and_preprocess(filepath):
    """Load NIfTI and preprocess: crop + percentile normalize."""
    vol = nib.load(filepath).get_fdata().astype(np.float32)
    
    # Crop foreground (non-zero bounding box + 5 voxel margin)
    nonzero = np.argwhere(vol > 0)
    if len(nonzero) == 0:
        return vol
    mins = np.maximum(nonzero.min(axis=0) - 5, 0)
    maxs = np.minimum(nonzero.max(axis=0) + 5, vol.shape)
    vol = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    
    # Percentile normalize to [0, 1]
    p1, p99 = np.percentile(vol[vol > 0], [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)
    
    return vol

####### EDIT HERE ###################################################
def create_model(model_type, model_name, weights_path, feat_dim):
    """Create model based on type."""
    if model_type == "dinov3":
        model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().cuda()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg), is_training=False)
    elif model_type == "usf-mae":
        model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="token")
        weights = torch.load(weights_path) # map_location and weights_only might need to be specified
        filtered_weights = {
            key:value
            for key, value in weights.items()
            if not key.startswith("decoder") and key != "mask_token"
        }
        model.load_state_dict(filtered_weights)
        model = model.eval().cuda()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg), is_training=False)
    # try to build ultrasam using timm samvit_base_patch16.sa1b
    elif model_type == "ultrasam":
        model = MODELS.build(ultrasam_config).cuda()
        model.eval()
        transform = TRANSFORMS.build(
            dict(
                type="FixScaleResize",
                scale=(1024, 1024),
                keep_ratio=True,
            )
        )

    #define model and transform for each modeltype. transform is just a callable which takes one image as output and returns the preprocessed torch.tensor 
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model, transform


def extract_slice_features(vol, model, transform, batch_size=256, model_provider="timm", axis=1, max_slices=200, mmpretrain=False):
    """Extract features for all slices along axis specified."""
    D = vol.shape[axis]

    # Determine middle slice range
    n = min(D, max_slices)
    start_idx = (D - n) // 2
    end_idx = start_idx + n
    indices = range(start_idx, end_idx)

    all_feats = []

    # Prepare middle slices as PIL images
    slices = []
    for i in indices:
        if axis == 0: # [H, W], float32 [0,1]
            s = vol[i]
        elif axis == 1:
            s = vol[:,i,:]  
        else:
            s = vol[:,:,i]
        
        s = (s * 255).astype(np.uint8)
        s = np.stack([s, s, s], axis=-1)  # [H, W, 3]

        if mmpretrain:
            res = {
                "img": s
            }
            res = transform(res)
            img = res["img"]
            slices.append(torch.from_numpy(img).permute(2,0,1))
        else:
            slices.append(Image.fromarray(s))

    # Batch transform and forward
    for start in range(0, len(slices), batch_size):
        batch_pil = slices[start:start + batch_size]
        if mmpretrain:
            batch_tensors = torch.stack(batch_pil).float().cuda()
        else:
            batch_tensors = torch.stack([transform(img) for img in batch_pil])
            batch_tensors = batch_tensors.cuda()

        with torch.no_grad():
            feats = model(batch_tensors)  # [B, feat_dim]

        if mmpretrain:
            all_feats.append(feats[0].cpu())
        else:
            all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)  # [D, feat_dim]

def resize_and_pad(batch, target_size):
    h,w = batch.shape[-2:]

    print(f"Original size: {batch.shape}")

    scale = target_size / max(h,w)
    new_h = round(h * scale)
    new_w = round(w * scale)

    batch = F.interpolate(batch.cuda(), size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    batch = F.pad(batch.cuda(), (pad_left, pad_right, pad_top, pad_bottom), value=0)

    print(f"Size after resize: {batch.shape}")

    return batch

def extract_slice_features_fast(vol, model, batch_size=64, img_size=1024, axis=1):
    D = vol.shape[axis]
    all_feats = []
    
    # Precompute all slices as tensors on GPU
    vol_tensor = torch.from_numpy(vol).float()  # [D, H, W]
    
    for start in range(0, D, batch_size):
        if axis == 0:
            batch = vol_tensor[start:start + batch_size]  # [B, H, W]
        elif axis == 1:
            batch = vol[:,start:start + batch_size,:]  
        else:
            batch = vol[:,:,start:start + batch_size]
        batch = batch.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
        # Resize on GPU
        batch = resize_and_pad(batch, img_size)
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        batch = (batch - mean) / std
        
        with torch.no_grad():
            feats = model(batch)[0]

        all_feats.append(feats.cpu())
    
    return torch.cat(all_feats)

def aggregate_features(sf, feat_dim):
    """Aggregate slice-level features to volume-level."""
    mean_feat = sf.mean(dim=0).numpy()
    max_feat = sf.max(dim=0).values.numpy()

    # Stacked: interpolate to fixed depth
    sf_interp = sf.unsqueeze(0).permute(0, 2, 1)  # [1, feat_dim, D]
    sf_interp = F.interpolate(sf_interp, size=N_SLICES, mode="linear", align_corners=False)
    stacked_feat = sf_interp.squeeze().permute(1, 0).flatten().numpy()

    return mean_feat, max_feat, stacked_feat

def find_nii_path(image_id, nii_base_path):
    """Find NIfTI file containing zero-padded image_id in its filename."""
    # Zero-pad to 6 digits (matching filename pattern)
    img_id_str = str(image_id).zfill(6)
    # List all .nii files in the directory
    for fname in os.listdir(nii_base_path):
        if fname.endswith('.nii.gz') and img_id_str in fname:
            return os.path.join(nii_base_path, fname)
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract features from a single model')
    parser.add_argument('--model', type=str, required=False,
                        help='Model name to extract features from')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if features already exist')
    parser.add_argument('--axis', type=int, required=False, help='Axis to slice the volume', default=1)
    args = parser.parse_args()
    
    
    # Validate model
    if args.model not in FE_MODELS:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {list(FE_MODELS.keys())}")
        return
    
    print("=" * 60)
    print(f"Feature Extraction: {args.model}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load mapping file
    print(f"\nLoading mapping file: {MAPPING_FILE}")
    df = pd.read_csv(MAPPING_FILE)
    print(f"Total images in CSV: {len(df)}")
    
    # Remove rows with NaN labels
    df = df.dropna(subset=[LABEL_COL])
    print(f"After removing NaN labels: {len(df)} images")
    
    # Keep ALL volumes (no patient deduplication)
    print(f"Processing all {len(df)} volumes")
    
    # Get image_ids from mapping file
    image_ids = df['image_id'].tolist()

    # Find paths for all image_ids (returns None if not found)
    selected_files = [find_nii_path(img_id, NII_PATH) for img_id in image_ids]

       # Filter to only existing files (non-None paths)
    valid_selected = [f for f in selected_files if f is not None]
    print(f"Valid NIfTI files: {len(valid_selected)}")
    if len(valid_selected) == 0:
        print("ERROR: No valid files found!")
        return
    
    # Get valid indices to extract labels/patient_ids
    valid_indices = [i for i, f in enumerate(selected_files) if f is not None]
    patient_ids = df.iloc[valid_indices]['record_id'].values
    image_ids = df.iloc[valid_indices]['image_id'].values
    
    # Check if skip existing
    if args.skip_existing:
        model_short = args.model.replace('.', '_')
        feature_file = os.path.join(FEATURE_DIR, f'{model_short}_features.npz')
        if os.path.exists(feature_file):
            print(f"\n*** Skipping {args.model} (features already exist) ***")
            return
    
    # Phase 1: Load all volumes into RAM
    print(f"\nLoading {len(valid_selected)} volumes into RAM...")
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=32) as executor:
        volumes = list(executor.map(load_and_preprocess, valid_selected))
    
    print(f"Loaded {len(volumes)} volumes in {time.time() - t0:.2f}s")
    
    # Get model config
    model_name, model_provider, weights_path, feat_dim = FE_MODELS[args.model]
    
    # Phase 2: Extract features
    print(f"\n{'='*60}")
    print(f"Model: {args.model} ({model_name})")
    print("=" * 60)
    
    # Create model
    model, transform = create_model(args.model, model_name, weights_path, feat_dim)
    
    # Verify feature dim
    with torch.no_grad():
        if model_provider == "timm":
            test_input = torch.randn(1, *resolve_data_config(model.pretrained_cfg)['input_size']).cuda()
            actual_feat_dim = model(test_input).shape[1]
        else:
            test_input = torch.randn(3, 3, 1024, 1024).cuda()
            actual_feat_dim = model(test_input)[0].shape
    print(f"Feature dimension: {actual_feat_dim}")
     
    # Extract features for all volumes
    print(f"Extracting features...")
    t0 = time.time()
    
    all_slice_features = []
    for vol in tqdm.tqdm(volumes):
        if transform != None:
            if args.model == "ultrasam":
                sf = extract_slice_features(vol, model, transform, batch_size=16, axis=args.axis, mmpretrain=True)
            else:
                sf = extract_slice_features(vol, model, transform, batch_size=BATCH_SIZE, axis=args.axis)
        else:
            sf = extract_slice_features_fast(vol, model, axis=args.axis)
        all_slice_features.append(sf)
    
    print(f"Feature extraction completed in {time.time() - t0:.2f}s")
    
    # Aggregate features
    print(f"Aggregating features...")
    
    mean_features = []
    max_features = []
    stacked_features_fixed_len = []
    
    for sf in all_slice_features:
        mean_f, max_f, stacked_f = aggregate_features(sf, actual_feat_dim)
        mean_features.append(mean_f)
        max_features.append(max_f)
        stacked_features_fixed_len.append(stacked_f)
    
    mean_features = np.stack(mean_features)
    max_features = np.stack(max_features)
    stacked_features_fixed_len = np.stack(stacked_features_fixed_len)
    stacked_features = np.stack(all_slice_features)
    
    print(f"  mean: {mean_features.shape}")
    print(f"  max: {max_features.shape}")
    print(f"  stacked fixed length: {stacked_features_fixed_len.shape}")
    print(f"  stacked: {stacked_features.shape}")

    # Save features to disk
    os.makedirs(FEATURE_DIR, exist_ok=True)
    model_short = args.model.replace('.', '_')
    output_path = os.path.join(FEATURE_DIR, f'{model_short}_view{args.axis}_features_v1.npz')
    
    np.savez(
        output_path,
        mean=mean_features,
        max=max_features,
        stacked_fixed_length=stacked_features_fixed_len,
        stacked=stacked_features,
        patient_ids=patient_ids,
        image_ids=image_ids
    )
    print(f"  Saved features to {output_path}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"Features saved to: {FEATURE_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()