import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

job_id = os.getenv("SLURM_JOB_ID", "dev")
musa = "irregular jz"
training_pos_path = f"/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/reports/840578/{musa}/train/posterier/fold4.csv"
validation_pos_path = f"/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/reports/840578/{musa}/validation/posterier/fold4.csv"
MAPPING_FILE = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/images_with_labels_nov25.csv'


training_pos = pd.read_csv(training_pos_path)
validation_pos = pd.read_csv(validation_pos_path)
posteriers = pd.concat([training_pos, validation_pos])

df = posteriers.groupby(["image_ids", "true"])["predicted prob"].apply(list).reset_index(name="posterier_list")
mapping = pd.read_csv(MAPPING_FILE)
df = df.merge(mapping[["image_id", "record_id"]], left_on="image_ids", right_on="image_id", how="left")
df = df.rename(columns={"record_id":"patient_ids"})
df["mean"] = df["posterier_list"].apply(np.mean)

outdir = Path(f"reports/{job_id}/{musa}/figures")
outdir.mkdir(parents=True, exist_ok=True)
patient_count = df["patient_ids"].value_counts()

plt.figure()
plot = patient_count.value_counts().sort_index().plot(kind="bar")
plot.bar_label(plot.containers[0])
plt.xlabel("Number of scans per patient")
plt.ylabel("Number of patients")
plt.title("Distribution of images per patient")
plt.savefig(f"{outdir}/distribution.png")