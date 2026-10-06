import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import get_data

job_id = os.getenv("SLURM_JOB_ID", "dev")
dev_set = True
ALL_MUSA = ['myometrial cysts','echogenic','globular','asymmetrical', 'fan shaped','irregular jz','interupted jz']
meta_data_path = "/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/data/meta_data_short.csv"

for musa in ALL_MUSA:
    training_pos_path = f"/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/reports/840578/{musa}/train/posterier/fold4.csv"
    validation_pos_path = f"/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/reports/840578/{musa}/validation/posterier/fold4.csv"
    
    outdir = Path(f"reports/{job_id}/{musa}/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    print(musa)

    meta_data_df = pd.read_csv(meta_data_path)
    training_pos = pd.read_csv(training_pos_path)
    validation_pos = pd.read_csv(validation_pos_path)
    posteriers = pd.concat([training_pos, validation_pos])

    df = posteriers.groupby(["image_ids", "true"])["predicted prob"].apply(list).reset_index(name="posterier_list")
    posteriers = np.stack(df["posterier_list"].values)
    dis = distance.cdist(posteriers, posteriers, "euclidean")
    dist_df = pd.DataFrame(dis, index=df["image_ids"], columns=df["image_ids"])
    df.rename(columns={'image_ids': 'image_id'}, inplace=True)
    df = df.merge(meta_data_df[['image_id','record_id', 'redcap_data_access_group', 'probe']], how='left', on='image_id')

    # plot distance matrix
    # plt.figure()
    # plt.imshow(dist_df, cmap='hot')
    # plt.savefig(f"{outdir}/dist_matrix_{musa}.png")
    # plt.close()

    per_plex = 100
    tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=42, perplexity=per_plex)
    embedding = tsne.fit_transform(dis)

    training_len = int(len(training_pos) / 200)

    for embedding_label in ["redcap_data_access_group", "probe", "true"]:
        plt.figure() 

        for label in set(df[embedding_label]):
            print(f"Label: {label}")
            
            if dev_set:
                cond = df[embedding_label] == label
                embeddings_to_use = embedding[cond]
            else: 
                df_to_use = df[training_len:]
                cond = df_to_use[embedding_label] == label
                embeddings_to_use = embedding[training_len:]
                embeddings_to_use = embeddings_to_use[cond]

            print(f"length of data from this hospital: {len(embeddings_to_use)}")
            plt.scatter(embeddings_to_use[:,0], embeddings_to_use[:,1], label=label)

        plt.title(f"tsne perplexity={per_plex}, label={embedding_label}")
        plt.legend()
        plt.xlabel("tsne 1")
        plt.ylabel("tsne 2")
        plt.savefig(f"{outdir}/embedding_{musa}_{embedding_label}.png")
        plt.close()



