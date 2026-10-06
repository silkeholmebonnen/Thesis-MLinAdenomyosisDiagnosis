import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import get_data

job_id = os.getenv("SLURM_JOB_ID", "dev")
ALL_MUSA = ['myometrial cysts','echogenic','globular','asymmetrical', 'fan shaped','irregular jz','interupted jz']
meta_data_path = "/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/data/meta_data_short.csv"

outdir = Path(f"reports/{job_id}/figures")
outdir.mkdir(parents=True, exist_ok=True)

meta_data_df = pd.read_csv(meta_data_path)

feature_ex_model = "ultrasam"
view = 1
version = "v1"
feature_path = f"/dcai/projects/cu_0014/data/copl/ultrasound_ml/preextracted_features/{feature_ex_model}_view{view}_features_{version}.npz"
features, labels, patient_ids, image_ids = get_data("stacked", feature_path)
features = np.reshape(features, (1109, 200*768))
df = pd.DataFrame(labels)
df.insert(0, "image_id", image_ids)
df.insert(1, "patient_id", patient_ids)
df = df.merge(meta_data_df, how='left', on='image_id')

print(df.head())
dis = distance.cdist(features, features, "euclidean")
dist_df = pd.DataFrame(dis, index=df["image_id"], columns=df["image_id"])

per_plex = 30
tsne = TSNE(n_components=2, metric="precomputed", init="random", random_state=42, perplexity=per_plex)
embedding = tsne.fit_transform(dis)

df["depth_cm"] = (
    df["depth_label"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)
df["depth_group"] = np.floor(df["depth_cm"]).astype("Int64").astype(str) + " cm"

bins = [0, 15, 20, 25, 30, np.inf]
labels = ["<=15", "16-20", "21-25", "26-30", ">30"]
df["frame_rate_group"] = pd.cut(df["frame_rate_hz"], bins=bins, labels=labels).astype(str)

for embedding_label in ["redcap_data_access_group", "probe", "thermal_index","depth_group","frame_rate_group","mechanical_index","preset"] + ALL_MUSA:
    plt.figure() 

    for label in set(df[embedding_label]):
        print(f"Label: {label}")
        
        cond = df[embedding_label] == label
        embeddings_to_use = embedding[cond]

        print(f"length of data from this hospital: {len(embeddings_to_use)}")
        plt.scatter(embeddings_to_use[:,0], embeddings_to_use[:,1], label=label)

    plt.title(f"tsne perplexity={per_plex}, label={embedding_label}")
    plt.legend()
    plt.xlabel("tsne 1")
    plt.ylabel("tsne 2")
    plt.savefig(f"{outdir}/embedding_{embedding_label}.png")
    plt.close()



