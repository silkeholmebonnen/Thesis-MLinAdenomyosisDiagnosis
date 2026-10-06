import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

filepath_to_scans = "/dcai/projects/cu_0014/data/copl/ultrasound_ml/kretz/nifti_kretz_converted2/"
scans = ["00095210_cartesian.nii.gz"]

for scan in scans:
    filepath = filepath_to_scans + scan

    vol = nib.load(filepath).get_fdata().astype(np.float32)

    print(vol.shape)

    for j in range(0,3):
        D = vol.shape[j]
        outdir = Path(f"plots/{scan}/view{j}")
        outdir.mkdir(parents=True, exist_ok=True)
        for i in range(0,D,20): 
            if j == 0:
                slice = plt.imshow(vol[i,:,:])
            elif j == 1:
                slice = plt.imshow(vol[:,i,:])
            else:
                slice = plt.imshow(vol[:,:,i])
            plt.savefig(f"{outdir}/slice{i}.png")

    # for j in range(1,2):
    #     D = vol.shape[j]
    #     start_idx = (D - 200) // 2
    #     end_idx = start_idx + 200
    #     outdir = Path(f"plots/{scan}/view{j}_fine")
    #     outdir.mkdir(parents=True, exist_ok=True)
    #     for i in range(start_idx, end_idx): 
    #         if j == 0:
    #             slice = plt.imshow(vol[i,:,:])
    #         elif j == 1:
    #             slice = plt.imshow(vol[:,i,:])
    #         else:
    #             slice = plt.imshow(vol[:,:,i])
    #         plt.savefig(f"{outdir}/slice{i}.png")