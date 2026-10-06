import os
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, brier_score_loss

import time
from utils import get_data
import numpy as np
import argparse

ALL_MUSA = ['myometrial cysts','echogenic','globular','asymmetrical', 'fan shaped','irregular jz','interupted jz']
SELECTED_MUSA = ['irregular jz','globular']

def store_posteriers(labels_validation, validation_posteriers, labels_train, train_posteriers, image_ids_validation, image_ids_train, musa_feature, fold):
    
    validation_pos_and_labels = {}
    validation_pos_and_labels.update({"image_ids": image_ids_validation})
    validation_pos_and_labels.update({"true": labels_validation}) 
    validation_pos_and_labels.update({"predicted prob": validation_posteriers}) 
    job_id = os.getenv("SLURM_JOB_ID", "dev")
    outdir = Path(f"reports/{job_id}/{musa_feature}/validation/posterier")
    outdir.mkdir(parents=True, exist_ok=True)
    validation_df = pd.DataFrame(validation_pos_and_labels)
    validation_df.to_csv(f"{outdir}/fold{fold}.csv")

    train_pos_and_labels = {}
    train_pos_and_labels.update({"image_ids": image_ids_train})
    train_pos_and_labels.update({f"true": labels_train}) 
    train_pos_and_labels.update({f"predicted prob": train_posteriers}) 
    outdir = Path(f"reports/{job_id}/{musa_feature}/train/posterier")
    outdir.mkdir(parents=True, exist_ok=True)
    train_df = pd.DataFrame(train_pos_and_labels)
    train_df.to_csv(f"{outdir}/fold{fold}.csv")

def store_dev_metrics(res):
    job_id = os.getenv("SLURM_JOB_ID", "dev")

    for musa, metrics in res.items():
        validation_auc_per_fold, train_auc_per_fold, validation_brier_per_fold, train_brier_per_fold = metrics
        val_outdir = Path(f"reports/{job_id}/{musa}/validation")
        val_outdir.mkdir(parents=True, exist_ok=True)
        val = {"Mean AUC": [np.mean(validation_auc_per_fold)], "Std AUC": [np.std(validation_auc_per_fold)], "Mean brier": [np.mean(validation_brier_per_fold)], "Std brier": [np.std(validation_brier_per_fold)]}
        train_df = pd.DataFrame(val)
        train_df.to_csv(f"{val_outdir}/metrics.csv")

        train_outdir = Path(f"reports/{job_id}/{musa}/train")
        train_outdir.mkdir(parents=True, exist_ok=True)
        train = {"Mean AUC": [np.mean(train_auc_per_fold)], "Std AUC": [np.std(train_auc_per_fold)], "Mean brier": [np.mean(train_brier_per_fold)], "Std brier": [np.std(train_brier_per_fold)]}
        train_df = pd.DataFrame(train)
        train_df.to_csv(f"{train_outdir}/metrics.csv")

def store_meta_data(feature_ex_model, classification_model, feature_type, view, hospital, rf_max_depth=None):
    job_id = os.getenv("SLURM_JOB_ID", "dev")
    outdir = Path(f"reports/{job_id}/")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(f"{outdir}/meta_data.txt", "w") as f:
        f.write(f"Feature extraction model: {feature_ex_model}\n")
        f.write(f"Classification model: {classification_model}\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"View: {view}\n")
        f.write(f"Hospital: {hospital}\n")
        if classification_model == "random_forest":
            f.write(f"Max depth: {rf_max_depth}")



def run_cross_val(classification_model, features, labels, groups, image_ids, musa_feature, rf_n_estimators=100, rf_max_depth=10, store_pos=True, feature_type="independent"):
    cv = StratifiedGroupKFold(shuffle=True, random_state=42)
    if classification_model == "random_forest":
        model = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=rf_n_estimators, max_depth=rf_max_depth, n_jobs=32)
    else: 
        model = LogisticRegression(class_weight="balanced", random_state=42, n_jobs=32)

    t0 = time.time()
    validation_auc_per_fold = []
    train_auc_per_fold = []
    validation_brier_per_fold = []
    train_brier_per_fold = []
    fold = 0
    labels = labels.to_numpy()

    for train_i, validation_i in cv.split(features, labels, groups):
        features_train, features_validation = features[train_i], features[validation_i]
        labels_train, labels_validation = labels[train_i], labels[validation_i]
        image_ids_train, image_ids_validation = image_ids[train_i], image_ids[validation_i]

        model.fit(features_train, labels_train)

        validation_posteriers = model.predict_proba(features_validation)[:,1]
        train_posteriers = model.predict_proba(features_train)[:,1]

        if store_pos:
            store_posteriers(labels_validation, validation_posteriers, labels_train, train_posteriers, image_ids_validation, image_ids_train, musa_feature, fold)

        if feature_type == "independent":
            validation_posteriers = validation_posteriers.reshape(-1, 200).mean(axis=1)
            train_posteriers = train_posteriers.reshape(-1, 200).mean(axis=1)
            labels_validation = labels_validation.reshape(-1, 200).mean(axis=1)
            labels_train = labels_train.reshape(-1,200).mean(axis=1)


        validation_auc_per_fold.append(roc_auc_score(labels_validation, validation_posteriers))
        validation_brier_per_fold.append(brier_score_loss(labels_validation, validation_posteriers))
        train_auc_per_fold.append(roc_auc_score(labels_train, train_posteriers))
        train_brier_per_fold.append(brier_score_loss(labels_train, train_posteriers))

        fold +=1

    print("\n------------------------------")
    print(f"Trained model in {time.time() - t0:.2f}s")
    print(f"Musa feature: {musa_feature}")
    print(f"validation AUC: {np.mean(validation_auc_per_fold):.4f} ({np.std(validation_auc_per_fold):.4f})")
    print(f"train AUC: {np.mean(train_auc_per_fold):.4f} ({np.std(train_auc_per_fold):.4f})")
    print(f"validation brier loss: {np.mean(validation_brier_per_fold):.4f} ({np.std(validation_brier_per_fold):.4f})")
    print(f"train brier loss: {np.mean(train_brier_per_fold):.4f} ({np.std(train_brier_per_fold):.4f})")

    return validation_auc_per_fold, train_auc_per_fold, validation_brier_per_fold, train_brier_per_fold

def include_data_from_hospital(hospital, features, labels, patient_ids, image_ids):
    meta_data_path = "/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/data/meta_data.tsv"
    meta_data_df = pd.read_csv(meta_data_path, sep="\t")

    if hospital.startswith("exclude_"):
        _, hospital = hospital.split("_")
        hosp = meta_data_df[meta_data_df["redcap_data_access_group"] != hospital]
        hosp_patient_ids = hosp["record_id"]
        is_from_hosp = np.isin(patient_ids, hosp_patient_ids)
        features = features[is_from_hosp]
        labels = labels[is_from_hosp]
        patient_ids = patient_ids[is_from_hosp]
        image_ids = image_ids[is_from_hosp]

    elif hospital != "all":
        hosp = meta_data_df[meta_data_df["redcap_data_access_group"] == hospital]
        hosp_patient_ids = hosp["record_id"]
        is_from_hosp = np.isin(patient_ids, hosp_patient_ids)
        features = features[is_from_hosp]
        labels = labels[is_from_hosp]
        patient_ids = patient_ids[is_from_hosp]
        image_ids = image_ids[is_from_hosp]

    return features, labels, patient_ids, image_ids

def train_classifier(feature_type, musa_feature, classification_model, feature_ex_model, view, rf_n_estimators=100, rf_max_depth=10, store_metrics=True, store_posteriers=True, store_meta=True, hospital="all"):
    feature_path = f"/dcai/projects/cu_0014/data/copl/ultrasound_ml/preextracted_features/{feature_ex_model}_view{view}_features_v1.npz"

    feature_type_to_retrieve = feature_type

    if feature_type_to_retrieve != "mean" and feature_type_to_retrieve != "max":
        feature_type_to_retrieve = "stacked"

    features, labels, patient_ids, image_ids = get_data(feature_type_to_retrieve, feature_path)
    features, labels, patient_ids, image_ids = include_data_from_hospital(hospital, features, labels, patient_ids, image_ids)

    entries = features.shape[0]

    if feature_type == "stacked":
        features = np.reshape(features, (entries, 200*768))
    elif feature_type == "stack10":
        features = features[:,::10,:]
        entries = features.shape[0]
        features = np.reshape(features, (entries, 20*768))
    elif feature_type == "pca":
        features = np.reshape(features, (entries, 200*768))
        pca = PCA(n_components=1000)
        features = pca.fit_transform(features)
    elif feature_type == "independent":
        features = np.reshape(features, (entries*200, 768))
        patient_ids = np.repeat(patient_ids, 200)
        image_ids = np.repeat(image_ids, 200)
        labels = labels.iloc[np.repeat(np.arange(len(labels)), 200)]
    elif feature_type == "independent10":
        features = features[:,::10,:]
        entries = features.shape[0]
        features = np.reshape(features, (entries*20, 768))
        patient_ids = np.repeat(patient_ids, 20)
        image_ids = np.repeat(image_ids, 20)
        labels = labels.iloc[np.repeat(np.arange(len(labels)), 20)]

    print(f"Feature extraction model: {feature_ex_model}")
    print(f"Classification model: {classification_model}")
    print(f"Feature type: {feature_type}")
    print(f"View: {view}")
    print(f"Hospital: {hospital}")
    print(f"Feature path: {feature_path}")

    if store_meta:
        store_meta_data(feature_ex_model, classification_model, feature_type, view, rf_max_depth, hospital)

    results_per_musa_feature = {}
    if musa_feature == "all":
        for m in ALL_MUSA:
            l = labels[m]
            res = run_cross_val(classification_model, features, l, patient_ids, image_ids, musa_feature=m, rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth, store_pos=store_posteriers, feature_type=feature_type)
            results_per_musa_feature.update({m: res})
    elif musa_feature == "selected":
        for m in SELECTED_MUSA:
            l = labels[m]
            res = run_cross_val(classification_model, features, l, patient_ids, image_ids, musa_feature=m, rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth, store_pos=store_posteriers, feature_type=feature_type)
            results_per_musa_feature.update({m: res})
    else:
        labels = labels[musa_feature]
        res = run_cross_val(classification_model, features, labels, patient_ids, image_ids, musa_feature=musa_feature, rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth, store_pos=store_posteriers, feature_type=feature_type)
        results_per_musa_feature.update({musa_feature: res})
    
    if store_metrics:
        store_dev_metrics(results_per_musa_feature)

    return results_per_musa_feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_ex_model', type=str, required=True,
                        help='Model name to extract features from (usf-mae,dinov3)')
    parser.add_argument('--musa_feature', type=str, required=True,
                        help='Musa feature to predict, or all, or selected')
    parser.add_argument('--feature_type', type=str, required=True,
                        help='stacked, mean, max, stack10')
    parser.add_argument('--classification_model', type=str, required=True,
                        help='random_forest, logistic_regression')
    parser.add_argument('--view', type=int, default=1,
                        help='0,1,2')
    parser.add_argument('--hospital', type=str, default="all",
                        help='hvidovre, hillerd, herlev, odense')
    args = parser.parse_args()

    train_classifier(args.feature_type, args.musa_feature, args.classification_model, args.feature_ex_model, args.view, hospital=args.hospital)


if __name__=="__main__":
    main()