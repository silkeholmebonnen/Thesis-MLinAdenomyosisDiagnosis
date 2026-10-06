import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from utils import get_data

target = "globular"
LABEL_COL = ['myometrial cysts','hyperechogenic','echogenic','globular','asymmetrical','fan shaped','irregular jz','interupted jz']
meta_data_path = "/dcai/projects/cu_0014/analysis/copl/ultrasound_ml/masterthesis_silke/data/meta_data.tsv"

meta_data_df = pd.read_csv(meta_data_path, sep="\t")

# remove columns with more than 50% nan values
missing_frac = meta_data_df.isna().mean()
cols_to_keep = missing_frac[missing_frac <= 0.5].index.tolist()
meta_filtered = meta_data_df[cols_to_keep]
print(f"Dropped {len(missing_frac) - len(cols_to_keep)} columns out of {len(missing_frac)}, left: {len(cols_to_keep)}")

# merge training data with meta data file
_, labels, patient_ids, _ = get_data("stacked", "/dcai/projects/cu_0014/data/copl/ultrasound_ml/preextracted_features/dinov3_view1_features_v1.npz")
unique_patients = np.unique(patient_ids)
print(f"number of unique patients: {len(unique_patients)}")

labels["record_id"] = pd.Series(patient_ids)
merged_meta = meta_filtered.merge(labels, on="record_id", how="inner")
merged_meta = merged_meta.drop_duplicates(subset=["record_id"], keep="first")
print(f"shape of merged df: {merged_meta.shape}")

# get categorical and numerical columns and remove ids and label columns
merged_without_ids_and_labels = merged_meta.drop(["record_id", "copl_num", "copl_num_1", "copl_num_2", "copl_num_3", "copl_num_4", "COPLID", "hash_id_w", "RunningIndex", "hash_id_m"], axis=1, errors="ignore")
merged_without_ids_and_labels = merged_without_ids_and_labels.drop(LABEL_COL, axis=1, errors="ignore")
numeric_cols = merged_without_ids_and_labels.select_dtypes(include='number').columns.tolist()
categorical_cols = merged_without_ids_and_labels.select_dtypes(include=['object', 'category']).columns.tolist()

# all_cols = merged_without_ids_and_labels.columns.to_list()
# X = merged_without_ids_and_labels.copy()

# # find best correlating features
# is_discrete = []
# for col in all_cols:
#     if col in categorical_cols:
#         X[col] = LabelEncoder().fit_transform(X[col].astype(str))
#         is_discrete.append(True)
#     else:
#         X[col] = X[col].fillna(X[col].median())
#         is_discrete.append(False)

# mi_scores = mutual_info_classif(X, merged_meta[target], discrete_features=is_discrete, random_state=0)
# mi_results = pd.Series(mi_scores, index=all_cols).sort_values(ascending=False)
# target_entropy = entropy(merged_meta[target].value_counts(normalize=True))
# mi_results = mi_results / target_entropy
# summary = pd.DataFrame({'mutual_info': mi_results}).sort_values('mutual_info', ascending=False)

# # make model based on the most correlating features
# top_20_features = mi_results.head(20).index.tolist() 
X = merged_without_ids_and_labels.copy()
y = merged_meta[target]

numeric_features = [c for c in X if c in numeric_cols]
categorical_features = [c for c in X if c in categorical_cols]
X[numeric_features] = SimpleImputer(strategy='median').fit_transform(X[numeric_features])
if categorical_features:
    X[categorical_features] = SimpleImputer(strategy='most_frequent').fit_transform(X[categorical_features])

X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    max_depth=10,
    class_weight='balanced', 
    random_state=42,
)
rf.fit(X_train, y_train)

# print auc
y_proba = rf.predict_proba(X_test)[:, 1] 
auc = roc_auc_score(y_test, y_proba)
print(f"target: {target}")
print(f"AUC: {auc:.3f}\n")

# print(summary.head(20))  
# for i in top_20_features:
#     print(i)
#     print(merged_meta[i].head())