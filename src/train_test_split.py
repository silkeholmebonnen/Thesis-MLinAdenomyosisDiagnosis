"""
File used to create the test_image_ids.csv
"""

import os
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

MAPPING_FILE = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/images_with_labels_nov25.csv'
NII_PATH = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/kretz/nifti_kretz_converted2'

LABEL_COL = ['myometrial cysts','hyperechogenic','echogenic','asymmetrical','globular', 'fan shaped','irregular jz','interupted jz']

TEST_SIZE = 0.1

def find_nii_path(image_id, nii_base_path):
    """Find NIfTI file containing zero-padded image_id in its filename."""
    # Zero-pad to 6 digits (matching filename pattern)
    img_id_str = str(image_id).zfill(6)
    # List all .nii files in the directory
    for fname in os.listdir(nii_base_path):
        if fname.endswith('.nii.gz') and img_id_str in fname:
            return os.path.join(nii_base_path, fname)
    return None


df = pd.read_csv(MAPPING_FILE)
print(f"Total images in CSV: {len(df)}")

df = df.dropna(subset=LABEL_COL)
print(f"After removing NaN labels: {len(df)} images")

image_ids = df['image_id'].tolist()
selected_files = [find_nii_path(img_id, NII_PATH) for img_id in image_ids]

valid_indices = [i for i, f in enumerate(selected_files) if f is not None]
labels = df.iloc[valid_indices][LABEL_COL].values
patient_ids = df.iloc[valid_indices]['record_id'].values
image_ids = df.iloc[valid_indices]['image_id'].values

unique_patients = np.unique(patient_ids)
print(f"number of unique patients: {len(unique_patients)}")

patient_labels = []

for patient in unique_patients:
    mask = patient_ids == patient
    labels_for_patient = labels[mask]

    # a patients labels are added together so it gets one if it has any image with that musa feature present
    patient_label = labels_for_patient.max(axis=0)

    patient_labels.append(patient_label)

patient_labels = np.array(patient_labels)

splitter = MultilabelStratifiedShuffleSplit(n_splits=1,test_size=TEST_SIZE, random_state=42)

train_patient_idx, test_patient_idx = next(splitter.split(unique_patients, patient_labels))

train_patient_ids = unique_patients[train_patient_idx]
test_patient_ids = unique_patients[test_patient_idx]

train_mask = np.isin(patient_ids, train_patient_ids)
test_mask = np.isin(patient_ids, test_patient_ids)

train_images = image_ids[train_mask]
test_images = image_ids[test_mask]

train_labels = labels[train_mask]
test_labels = labels[test_mask]

print(f"Train and test is disjoint: {set(train_patient_ids).isdisjoint(set(test_patient_ids))}")

print(f"Overall distribution: {labels.mean(axis=0)}")
print(f"Train: {train_labels.mean(axis=0)}")
print(f"Test: {test_labels.mean(axis=0)}")

print(len(test_images))

df = pd.DataFrame({"image_id": test_images})

df.to_csv('/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/src/test_images/test_image_ids.csv', index=False)