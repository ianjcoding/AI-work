# filename: semi_supervised_food_insecurity.py
# Requirements: scikit-learn, pandas, numpy, matplotlib, joblib
import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)  

# Generate fake dataset for testing
np.random.seed(42)
num_regions = 200
num_features = 10

features = np.random.randn(num_regions, num_features)
labels = np.random.choice([0, 1, np.nan], size=num_regions, p=[0.3, 0.3, 0.4])
region_ids = [f"Region-{i}" for i in range(num_regions)]

df_fake = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(num_features)])
df_fake["region_id"] = region_ids
df_fake["label"] = labels

df_fake.to_csv("data/regions_features.csv", index=False)

print("Generated sample dataset at: data/regions_features.csv")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib

# ---------- Load your merged per-region dataset ----------
# Expected: df with features and column "label" where label is 0/1 for AtRisk/NotAtRisk
# Unlabeled rows should have label = np.nan or -1.
df = pd.read_csv("data/regions_features.csv")  # replace with your path

# ---------- Preprocessing ----------
features = [c for c in df.columns if c not in ("region_id", "label")]
X_raw = df[features].copy()
y_raw = df["label"].copy()  # contains some NaNs for unlabeled

# Impute and scale
imputer = SimpleImputer(strategy="median")
X_imp = imputer.fit_transform(X_raw)
scaler = StandardScaler()
X = scaler.fit_transform(X_imp)

# ---------- Train/test split (keep honest test set) ----------
# Keep a hold-out test set of fully labeled rows only
labeled_mask = ~y_raw.isna()
X_labeled = X[labeled_mask]
y_labeled = y_raw[labeled_mask].astype(int)
X_unlabeled = X[~labeled_mask]

# Split labeled into train_labeled and test (use stratify if possible)
X_train_lab, X_test, y_train_lab, y_test = train_test_split(
    X_labeled, y_labeled, test_size=0.25, random_state=42, stratify=y_labeled
)

# Create semi-supervised training pool: small labeled set + many unlabeled
# Option: further reduce X_train_lab to simulate few labels
n_labeled_small = max(20, int(0.10 * len(X_train_lab)))  # e.g., 10% or at least 20
indices = np.arange(len(X_train_lab))
rng = np.random.RandomState(42)
rng.shuffle(indices)
small_idx = indices[:n_labeled_small]
X_small_lab = X_train_lab[small_idx]
y_small_lab = y_train_lab.iloc[small_idx].values

# Combine small labeled and unlabeled pool
X_pool = np.vstack([X_small_lab, X_unlabeled])
# Labels: labeled entries keep 0/1, unlabeled set to -1 for sklearn semi_supervised
y_pool = np.concatenate([y_small_lab, -1 * np.ones(len(X_unlabeled), dtype=int)])

# ---------- Approach 1: Label Spreading ----------
label_spread = LabelSpreading(kernel='rbf', alpha=0.2, max_iter=1000)
label_spread.fit(X_pool, y_pool)
pseudo_labels = label_spread.transduction_  # labels for entire pool

# Grab indices that were unlabeled originally and have high confidence via label_spread.label_distributions_
# We can choose pseudo labels only where model is confident
probs = label_spread.label_distributions_
# For unlabeled pool entries (positions after n_labeled_small), choose those with max prob > thresh
thresh = 0.85
unlabeled_start = len(X_small_lab)
selected_mask = (np.max(probs[unlabeled_start:], axis=1) > thresh)
X_pseudo = X_unlabeled[selected_mask]
y_pseudo = pseudo_labels[unlabeled_start:][selected_mask]

# ---------- Approach 2: Self-training with RandomForest ----------
# Start with small labeled set, augment with pseudo labels from label spreading
X_self_train = np.vstack([X_small_lab, X_pseudo])
y_self_train = np.concatenate([y_small_lab, y_pseudo])

clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_self_train, y_self_train)

# Evaluate on hold-out test
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba)
}
print("Evaluation on test set:", metrics)

# Save models and preprocessing
os.makedirs("models", exist_ok=True)
joblib.dump({
    "imputer": imputer,
    "scaler": scaler,
    "label_spread": label_spread,
    "classifier": clf
}, "models/semi_supervised_pipeline.joblib")
