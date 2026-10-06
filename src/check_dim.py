import nibabel as nib
import numpy as np
from pathlib import Path

filepath_to_scans = "/dcai/projects/cu_0014/data/copl/ultrasound_ml/kretz/nifti_kretz_converted2/"

def load_and_preprocess(filepath):
    """Load NIfTI and preprocess: crop + percentile normalize."""
    vol = nib.load(filepath).get_fdata().astype(np.float32)

    nonzero = np.argwhere(vol > 0)
    if len(nonzero) == 0:
        return vol
    mins = np.maximum(nonzero.min(axis=0) - 5, 0)
    maxs = np.minimum(nonzero.max(axis=0) + 5, vol.shape)
    vol = vol[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    p1, p99 = np.percentile(vol[vol > 0], [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    return vol

scans = sorted(Path(filepath_to_scans).glob("*.nii.gz"))
print(f"Found {len(scans)} files")

shapes = []
failed = []

for filepath in scans:
    try:
        vol = load_and_preprocess(filepath)
        if vol.size == 0 or vol.ndim != 3:
            print(f"WARNING: {filepath.name} produced an unexpected volume, shape={vol.shape}")
            failed.append(filepath.name)
            continue
        shapes.append(vol.shape)
        print(f"{filepath.name}: {vol.shape}")
    except Exception as e:
        print(f"FAILED on {filepath.name}: {e}")
        failed.append(filepath.name)

shapes = np.array(shapes) 

print(f"Successfully processed: {len(shapes)} / {len(scans)}")

for axis in range(3):
    col = shapes[:, axis]
    print(f"dim {axis}: min={col.min()}, max={col.max()}, "
          f"mean={col.mean():.1f}, median={np.median(col):.1f}")
    print(f"  percentiles [5,25,50,75,90,95]: "
          f"{np.percentile(col, [5, 25, 50, 75, 90, 95])}")