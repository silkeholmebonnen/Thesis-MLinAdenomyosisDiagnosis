import pandas as pd
from utils import get_data
import json
import operator
import numpy as np

target = "irregular jz"
LABEL_COL = ['myometrial cysts','hyperechogenic','echogenic','globular','asymmetrical','fan shaped','irregular jz','interupted jz']

extra_meta = '/dcai/projects/cu_0014/data/copl/ultrasound_ml/kretz/nrrds_ketz_converted'

meta_data_path = "/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/data/meta_data.tsv"
meta_data_df = pd.read_csv(meta_data_path, sep="\t")

print(f"min age: {min(meta_data_df['merged_maternal_age_at_inclusion'])}")
print(f"max age: {max(meta_data_df['merged_maternal_age_at_inclusion'])}")
print(f"mean age: {meta_data_df['merged_maternal_age_at_inclusion'].mean()}")

_, labels, patient_ids, image_ids = get_data("stacked", "/dcai/projects/cu_0014/data/copl/ultrasound_ml/preextracted_features/dinov3_view1_features_v1.npz")

data = {
    'image_id': image_ids,
    'record_id': patient_ids
}
dev_data = pd.DataFrame(data)

merged_meta = dev_data.merge(meta_data_df[['record_id', 'redcap_data_access_group']], how='left', on='record_id')


hospitals = ["hvidovre", "hillerd", "herlev", "odense"]
for h in hospitals:
    hosp = merged_meta[merged_meta["redcap_data_access_group"] == h]
    print(f"{h} shape: {hosp.shape}")

    probes = []
    for inst in hosp['image_id']:
        inst = str(inst).zfill(8)
        path = f"{extra_meta}/{inst}_meta.json"
        with open(path, "r") as file:
            meta_json = json.load(file)
            probe = meta_json["acquisition"]["probe"]
            probes.append(probe)

    print(f"Hospital {h} used these probes: ")
    for p in set(probes):
        print(f"Probe: {p}")
        print(f"Count: {operator.countOf(probes, p)}")

depth = []
frame_rate = []
mi_index = []
present = []
probes = []
software = []
thermal = []
for i in image_ids:
    inst = str(i).zfill(8)
    path = f"{extra_meta}/{inst}_meta.json"
    with open(path, "r") as file:
        meta_json = json.load(file)
        depth.append(meta_json["acquisition"]["depth_label"])
        frame_rate.append(meta_json["acquisition"]["frame_rate_hz"])
        mi_index.append(meta_json["acquisition"]["mechanical_index"])
        present.append(meta_json["acquisition"]["preset"])
        probes.append(meta_json["acquisition"]["probe"])
        software.append(meta_json["acquisition"]["software"])
        thermal.append(meta_json["acquisition"]["thermal_index"])


merged_meta.insert(3, "probe", probes)
merged_meta.insert(4, "software", software)
merged_meta.insert(5, "thermal_index", thermal)
merged_meta.insert(6, "depth_label", depth)
merged_meta.insert(7, "frame_rate_hz", frame_rate)
merged_meta.insert(8, "mechanical_index", mi_index)
merged_meta.insert(9, "preset", present)

merged_meta.to_csv('data/meta_data_short.csv')
