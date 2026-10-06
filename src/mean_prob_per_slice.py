import os
from pathlib import Path
import numpy as np
import pandas as pd
from plot_pred_prob_per_slice import plot_and_save_with_std

job_id = os.getenv("SLURM_JOB_ID", "dev")
# musa = "globular"
job_to_find = 840765
ALL_MUSA = ['myometrial cysts','echogenic','globular','asymmetrical', 'fan shaped','irregular jz','interupted jz']
for musa in ALL_MUSA:
    training_pos_path = f"/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/reports/{job_to_find}/{musa}/train/posterier/fold4.csv"
    validation_pos_path = f"/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/reports/{job_to_find}/{musa}/validation/posterier/fold4.csv"
    MAPPING_FILE = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/images_with_labels_nov25.csv'

    print(musa)

    training_pos = pd.read_csv(training_pos_path)
    validation_pos = pd.read_csv(validation_pos_path)
    all_posteriers = pd.concat([training_pos, validation_pos])
    # posteriers = validation_pos

    for str_set, posteriers in [("dev", all_posteriers), ("val", validation_pos)]:

        df = posteriers.groupby(["image_ids", "true"])["predicted prob"].apply(list).reset_index(name="posterier_list")
        posteriers = np.stack(df["posterier_list"].values)
        mapping = pd.read_csv(MAPPING_FILE)
        df = df.merge(mapping[["image_id", "record_id"]], left_on="image_ids", right_on="image_id", how="left")
        df = df.rename(columns={"record_id":"patient_ids"})
        df["mean"] = df["posterier_list"].apply(np.mean)



        df_true = df[df["true"] == 1.0]
        df_false = df[df["true"] == 0.0]

        df_true = df_true.sort_values(by=["mean"])
        false_postives = np.array(df_true['posterier_list'].head(10).tolist()).mean(axis=0)
        false_postives_std = np.array(df_true['posterier_list'].head(10).tolist()).std(axis=0)
        plot_and_save_with_std(false_postives, false_postives_std, job_id, musa, f"FP_mean_{str_set}", f"Mean pred prob pr slice over 10 worst performing postives (1) in {str_set} set")
        true_postives = np.array(df_true['posterier_list'].tail(10).tolist()).mean(axis=0)
        true_postives_std = np.array(df_true['posterier_list'].tail(10).tolist()).std(axis=0)
        plot_and_save_with_std(true_postives, true_postives_std, job_id, musa, f"TP_mean_{str_set}", f"Mean pred prob pr slice over 10 best performing postives (1) in {str_set} set")
        arr_true = np.array(df_true['posterier_list'].tolist())
        all_positives= arr_true.mean(axis=0)
        all_positives_std = arr_true.std(axis=0)
        plot_and_save_with_std(all_positives, all_positives_std, job_id, musa, f"T_mean_{str_set}", f"Mean pred prob pr slice over all postives (1) in {str_set} set")

        df_false = df_false.sort_values(by=["mean"])
        arr_false = np.array(df_false['posterier_list'].tolist())

        true_negatives = np.array(df_false['posterier_list'].head(10).tolist()).mean(axis=0)
        true_negatives_std = np.array(df_false['posterier_list'].head(10).tolist()).std(axis=0)
        plot_and_save_with_std(true_negatives, true_negatives_std, job_id, musa, f"TN_mean_{str_set}", f"Mean predprob pr slice over 10 best performing negatives (0) in {str_set} set")
        false_neatives = np.array(df_false['posterier_list'].tail(10).tolist()).mean(axis=0)
        false_neatives_std = np.array(df_false['posterier_list'].tail(10).tolist()).std(axis=0)
        plot_and_save_with_std(false_neatives, false_neatives_std, job_id, musa, f"FN_mean_{str_set}", f"Mean pred prob pr slice over 10 worst performing negatives (0) in {str_set} set")
        all_negatives = arr_false.mean(axis=0)
        all_negatives_std = arr_false.std(axis=0)
        plot_and_save_with_std(all_negatives, all_negatives_std, job_id, musa, f"N_mean_{str_set}", f"Mean pred prob pr slice over all negatives (0) in {str_set} set")

