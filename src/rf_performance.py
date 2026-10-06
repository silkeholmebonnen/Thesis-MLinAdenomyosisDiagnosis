
import os
from pathlib import Path
from utils import get_data
from classifier import train_classifier
import numpy as np
import argparse
import matplotlib.pyplot as plt

SELECTED_MUSA = ['irregular jz','globular']


parser = argparse.ArgumentParser()
parser.add_argument('--feature_ex_model', type=str, required=True,
                    help='Model name to extract features from (usf-mae,dinov3)')
parser.add_argument('--feature_type', type=str, required=True,
                    help='stacked, mean, max, stack10')
parser.add_argument('--view', type=int, default=1,
                    help='0,1,2')
args = parser.parse_args()

job_id = os.getenv("SLURM_JOB_ID", "dev")
max_depths = [2, 4, 6, 7, 8, 9, 10, 15, 20, 30]

for musa in ['irregular jz', 'fan shaped']:
    
    plt.figure() 
    plt.title(f"AUC and max depth of tree for musa feature {musa}")

    mean_validation_auc_per_depth = []
    mean_train_auc_per_depth = []
    std_validation_auc_per_depth = []
    std_train_auc_per_depth = []

    for max_depth in max_depths:
        res_dict = train_classifier(args.feature_type, musa, "random_forest", args.feature_ex_model, args.view, rf_max_depth=max_depth, store_posteriers=False)
        validation_aucs, train_aucs, _, _ = res_dict.get(musa)
        mean_validation_auc_per_depth.append(np.mean(validation_aucs))
        mean_train_auc_per_depth.append(np.mean(train_aucs))
        std_validation_auc_per_depth.append(np.std(validation_aucs))
        std_train_auc_per_depth.append(np.std(train_aucs))

    mean_validation_auc_per_depth = np.array(mean_validation_auc_per_depth)
    mean_train_auc_per_depth = np.array(mean_train_auc_per_depth)
    std_validation_auc_per_depth = np.array(std_validation_auc_per_depth)
    std_train_auc_per_depth = np.array(std_train_auc_per_depth)
            
    plt.plot(max_depths, mean_validation_auc_per_depth, label=f"Validation AUC")
    plt.fill_between(
        max_depths,
        mean_validation_auc_per_depth - std_validation_auc_per_depth,
        mean_validation_auc_per_depth + std_validation_auc_per_depth,
        alpha=0.3,
        label="±1 std val AUC"
    )
    plt.plot(max_depths, mean_train_auc_per_depth, label=f"Train AUC")
    plt.fill_between(
        max_depths,
        mean_train_auc_per_depth - std_train_auc_per_depth,
        mean_train_auc_per_depth + std_train_auc_per_depth,
        alpha=0.3,
        label="±1 std train AUC"
    )
    plt.legend()
    plt.xlabel("Max depth")
    plt.ylabel("AUC")
    outdir = Path(f"reports/{job_id}/{musa}/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{outdir}/maxdepthfig.png")
    plt.close()