
import numpy as np
import pandas as pd


MAPPING_FILE = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/images_with_labels_nov25.csv'
LABEL_COL = ['myometrial cysts','hyperechogenic','echogenic','globular','asymmetrical','fan shaped','irregular jz','interupted jz']


def get_data(data_type, feature_filepath, development=True):
    data = np.load(feature_filepath)

    test_reserved_img_ids = np.genfromtxt(
        '/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/data/test_image_ids.csv',
        skip_header=1
    )

    mapping = pd.read_csv(MAPPING_FILE)
    mapping = mapping.dropna(subset=LABEL_COL)

    correct_label_mask = np.isin(data["image_ids"], mapping['image_id'])
    image_ids =  data["image_ids"][correct_label_mask]
    features = data[data_type][correct_label_mask]
    patient_ids = data["patient_ids"][correct_label_mask]

    # print(f"length of image_ids before removing: {len(image_ids)}")

    test_reserved_img_ids = test_reserved_img_ids.astype(image_ids.dtype)

    # Boolean mask: True = keep, False = drop (reserved for test)
    is_test = np.isin(image_ids, test_reserved_img_ids)
    if development:
        keep_mask = ~is_test
    else: 
        keep_mask = is_test

    development_image_ids = image_ids[keep_mask]
    features = features[keep_mask]        
    test_reserved_patient_ids = patient_ids[~keep_mask]
    patient_ids = patient_ids[keep_mask]

   # print(f"length of image_ids after removing (should be 122 less than before): {len(development_image_ids)}")

    n_matched = is_test.sum()
    if n_matched != len(test_reserved_img_ids):
        print(f"WARNING: expected to remove {len(test_reserved_img_ids)} ids, "
              f"but only matched {n_matched} in image_ids. Check dtypes/values.")

    # print(f"Train and test image ids is disjoint: "
    #       f"{set(development_image_ids.tolist()).isdisjoint(set(test_reserved_img_ids.tolist()))}")
    
    # print(f"Train and test patient ids is disjoint: "
    #     f"{set(patient_ids.tolist()).isdisjoint(set(test_reserved_patient_ids.tolist()))}")


    labels = mapping[mapping['image_id'].isin(development_image_ids)][LABEL_COL]

    # print(f"shape of features: {features.shape}")
    # print(f"shape of labels: {labels.shape}")
    # print(f"shape of patient: {patient_ids.shape}")

    return features, labels, patient_ids, development_image_ids

def get_data_all_slices(feature_filepath, development=True):
    data = np.load(feature_filepath)

    test_reserved_img_ids = np.genfromtxt(
        '/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/data/test_image_ids.csv',
        skip_header=1
    )

    mapping = pd.read_csv(MAPPING_FILE)
    mapping = mapping.dropna(subset=LABEL_COL)

    correct_label_mask = np.isin(data["image_ids"], mapping['image_id'])
    image_ids =  data["image_ids"][correct_label_mask]
    patient_ids = data["patient_ids"][correct_label_mask]
    flat = data["stacked_flat"]      # [total_slices, 768]
    offsets = data["offsets"]        # [1110]
    all_feats = [flat[offsets[i]:offsets[i+1]] for i in range(len(offsets) - 1)]
    feats_by_image = {pid: all_feats[i] for i, pid in enumerate(image_ids)}

    test_reserved_img_ids = test_reserved_img_ids.astype(image_ids.dtype)

    # Boolean mask: True = keep, False = drop (reserved for test)
    is_test = np.isin(image_ids, test_reserved_img_ids)
    if development:
        keep_mask = ~is_test
    else: 
        keep_mask = is_test

    development_image_ids = image_ids[keep_mask]     
    patient_ids = patient_ids[keep_mask]
    features = {iid: feats_by_image[iid] for iid in development_image_ids}

    n_matched = is_test.sum()
    if n_matched != len(test_reserved_img_ids):
        print(f"WARNING: expected to remove {len(test_reserved_img_ids)} ids, "
              f"but only matched {n_matched} in image_ids. Check dtypes/values.")


    labels = mapping[mapping['image_id'].isin(development_image_ids)][LABEL_COL]

    print(f"shape of features: {len(features.keys())}")
    print(f"shape of labels: {labels.shape}")
    print(f"shape of patient: {patient_ids.shape}")

    return features, labels, patient_ids, development_image_ids

# feature_path = f"/dcai/projects/cu_0014/data/copl/ultrasound_ml/preextracted_features/dinov3_view1_features_all.npz"

# features, labels, patient_ids, image_ids = get_data_all_slices(feature_path)