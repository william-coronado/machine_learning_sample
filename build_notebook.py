"""
build_notebook.py  — run once to generate ieee_cis_feature_engineering.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0",
    },
}

cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ============================================================
# 0. Title
# ============================================================
md("""\
# Advanced Feature Engineering — IEEE-CIS Fraud Detection

This notebook is a **senior data-science reference guide** to feature engineering
for tabular fraud detection. It demonstrates every major technique category
with concrete implementations and explains *why* each technique works.

## Dataset
The [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
dataset consists of:
- **Transaction table** — ~590 k rows × 394 columns
  (TransactionDT, TransactionAmt, ProductCD, card1–6, addr, C/D/M/V features)
- **Identity table** — ~144 k rows × 41 columns
  (DeviceType, DeviceInfo, id_01–id_38)

Because the dataset requires Kaggle authentication, this notebook uses a
**synthetic replica** (`load_synthetic_ieee`) that mirrors the real dataset's
schema, missing-value patterns, cardinalities, and class imbalance (~3.5 % fraud).
The same code runs unchanged on the real CSV files — just swap the loader.

## Techniques covered
| Section | Technique |
|---------|-----------|
| §3 | Missingness analysis & heuristic imputation |
| §4 | Frequency encoding & smoothed target (mean) encoding |
| §5 | Hand-crafted interaction features |
| §6 | Entity-level aggregation (group statistics) |
| §7 | Temporal feature extraction + cyclical sin/cos encoding |
| §8 | PCA compression of correlated V-feature blocks |
| §9 | Feature selection: variance, correlation pruning, permutation importance |
| §10 | Baseline vs. engineered-features model lift |
""")

# ============================================================
# 1. Setup & Imports
# ============================================================
md("## 1. Setup & Imports")

code("""\
import sys, os
sys.path.insert(0, os.path.abspath(".."))   # allow notebook to find the package

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# ── project helpers ──────────────────────────────────────────────────────────
from feature_engineering.preprocessing import (
    load_synthetic_ieee,
    reduce_mem_usage,
    summarise_missing,
    impute_by_type,
)
from feature_engineering.encoders import TargetEncoder, FrequencyEncoder
from feature_engineering.aggregations import GroupAggregator
from feature_engineering.temporal import extract_temporal_features, extract_email_features
from feature_engineering.selection import (
    drop_high_correlation,
    variance_threshold_select,
    importance_select,
    pca_compress,
)
from feature_engineering.utils import (
    plot_missing,
    plot_feature_importance,
    roc_comparison,
    plot_pca_explained_variance,
    evaluate_model,
)

sns.set_theme(style="whitegrid", palette="muted")
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", "{:.4f}".format)
print("All imports OK.")
""")

# ============================================================
# 2. Load Data
# ============================================================
md("""\
## 2. Load Synthetic IEEE-CIS Data

`load_synthetic_ieee` returns two DataFrames with the exact same schema as
the Kaggle competition files.
To use the **real** data, replace the cell below with:

```python
trans    = pd.read_csv("train_transaction.csv")
identity = pd.read_csv("train_identity.csv")
```
""")

code("""\
trans, identity = load_synthetic_ieee(n_rows=150_000, seed=42)

print(f"Transactions : {trans.shape[0]:,} rows × {trans.shape[1]} columns")
print(f"Identity     : {identity.shape[0]:,} rows × {identity.shape[1]} columns")
print(f"\\nFraud rate   : {trans['isFraud'].mean() * 100:.2f} %")
trans.head(3)
""")

code("""\
# Merge identity on TransactionID (left join — most approaches merge here)
df = trans.merge(identity, on="TransactionID", how="left")
print(f"Merged shape : {df.shape[0]:,} rows × {df.shape[1]} columns")

# Downcast dtypes immediately to reduce RAM
df = reduce_mem_usage(df)
""")

# ============================================================
# 3. EDA — overview
# ============================================================
md("""\
## 3. Exploratory Data Analysis

Before engineering features we need to understand:
1. Class imbalance
2. Missingness patterns
3. Distribution of the key numeric columns
""")

code("""\
# ── Class imbalance ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

fraud_counts = df["isFraud"].value_counts()
axes[0].bar(["Legit", "Fraud"], fraud_counts.values,
            color=["steelblue", "tomato"], edgecolor="white")
axes[0].set_title("Class distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(fraud_counts.values):
    axes[0].text(i, v + 200, f"{v:,}", ha="center", fontsize=10)

# TransactionAmt distribution (log scale) by class
legit = df.loc[df["isFraud"] == 0, "TransactionAmt"].clip(upper=5000)
fraud = df.loc[df["isFraud"] == 1, "TransactionAmt"].clip(upper=5000)
axes[1].hist(legit, bins=80, alpha=0.6, label="Legit", color="steelblue", density=True)
axes[1].hist(fraud, bins=80, alpha=0.6, label="Fraud", color="tomato", density=True)
axes[1].set_xlabel("TransactionAmt (clipped at 5 000)")
axes[1].set_title("Transaction amount distribution by class")
axes[1].legend()

plt.tight_layout()
plt.show()
""")

code("""\
# ── Missingness ──────────────────────────────────────────────────────────────
# Focus on columns with at least 1 % missing for readability
miss_summary = summarise_missing(df)
print("Top 20 columns by missingness:")
display(miss_summary[miss_summary["pct_missing"] > 1].head(20))

plot_missing(df, figsize=(11, 6), max_cols=35,
             title="Missing-value rates (top 35 columns)")
""")

code("""\
# ── Correlation heatmap (key numeric features only) ──────────────────────────
key_cols = ["TransactionAmt", "C1", "C2", "C5", "C6",
            "D1", "D2", "D4", "D10", "D15", "isFraud"]
key_cols = [c for c in key_cols if c in df.columns]

fig, ax = plt.subplots(figsize=(10, 8))
corr = df[key_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Pearson correlation — key numeric features")
plt.tight_layout()
plt.show()
""")

# ============================================================
# §3 — Missing values
# ============================================================
md("""\
## 4. Missing-Value Imputation

### Why is this critical for fraud detection?

The IEEE-CIS dataset contains **block-wise missingness** in the V-features:
columns within the same block are either all observed or all missing for a
given transaction. This means missingness itself is a strong signal —
a transaction with V1–V11 missing likely came from a different device
fingerprinting pathway than one with all V columns observed.

### Strategy: -999 sentinel for tree-based models

For gradient-boosted trees and random forests, the best practice is to fill
numeric NaNs with a sentinel value (`-999`) that lies far outside the
natural range. The tree can then create a split `feature < -998.5` to
distinguish missing from non-missing, effectively learning the missingness
signal automatically.

For linear models and neural networks, use **median** imputation + add a
binary `{col}_was_nan` indicator column.
""")

code("""\
# ── Add was_nan indicators for key partially-missing columns ─────────────────
# This preserves the missingness signal for linear models
high_miss_cols = (
    miss_summary[miss_summary["pct_missing"].between(5, 95)]
    .index[:20].tolist()
)
# Only numeric columns
high_miss_cols = [c for c in high_miss_cols
                  if c in df.select_dtypes(include=np.number).columns]

for col in high_miss_cols:
    df[f"{col}_was_nan"] = df[col].isna().astype(np.int8)

print(f"Added {len(high_miss_cols)} *_was_nan indicator columns.")

# ── Impute: -999 sentinel for trees ─────────────────────────────────────────
df = impute_by_type(df, cat_fill="Unknown", num_strategy="minus999")

print("\\nRemaining missing values after imputation:",
      df.isnull().sum().sum())
""")

# ============================================================
# §4 — Encoding
# ============================================================
md("""\
## 5. Categorical Encoding Strategies

Raw categorical columns like `card4` ("visa", "mastercard") and
`P_emaildomain` ("gmail.com", "anonymous.com") contain predictive
information that must be encoded numerically.

| Encoding | Best for | Risk |
|----------|----------|------|
| One-hot | Low-cardinality (<20 levels) | Dimensionality explosion for high-card |
| **Frequency** | High-cardinality, no target leakage | Loses within-category target info |
| **Target (mean)** | High-cardinality, strong signal | Leakage → use LOO or CV folds |
| Label | Trees only (preserves ordinality) | Ordinal assumption may be wrong |

### Frequency encoding
Replace each category with its fractional frequency.
`P_emaildomain = "gmail.com"` → 0.35 (35 % of transactions used Gmail).

### Smoothed target encoding with leave-one-out
Replace each category with the smoothed conditional fraud rate.
`P_emaildomain = "anonymous.com"` → 0.12 (12 % of anonymous.com transactions were fraud).

LOO: when encoding the *training set*, each row uses the group mean computed
**excluding itself**. This prevents the model from learning a trivially
perfect mapping `encoded_val == y`.
""")

code("""\
# Columns to encode
cat_cols = ["ProductCD", "card4", "card6",
            "P_emaildomain", "R_emaildomain",
            "DeviceType"]
cat_cols = [c for c in cat_cols if c in df.columns]

# ── Train / test split (before any fitting) ───────────────────────────────
# IMPORTANT: fit encoders ONLY on train to prevent leakage
y = df["isFraud"]
X = df.drop(columns=["isFraud", "TransactionID"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean():.4f}")
print(f"Test  fraud rate: {y_test.mean():.4f}")
""")

code("""\
# ── Frequency encoding ───────────────────────────────────────────────────────
freq_enc = FrequencyEncoder(cols=cat_cols)
X_train = freq_enc.fit_transform(X_train)   # fit on train only
X_test  = freq_enc.transform(X_test)

print("Frequency-encoded columns added:")
for col in cat_cols:
    sample = X_train[f"{col}_freq"].describe()
    print(f"  {col}_freq: mean={sample['mean']:.4f}, "
          f"min={sample['min']:.4f}, max={sample['max']:.4f}")
""")

code("""\
# ── Target encoding (LOO on train, smoothed mean on test) ───────────────────
high_card_cols = ["P_emaildomain", "R_emaildomain", "card4", "card6", "ProductCD"]
high_card_cols = [c for c in high_card_cols if c in X_train.columns]

te_enc = TargetEncoder(cols=high_card_cols, smoothing=20, noise_level=0.01)
X_train = te_enc.fit_transform(X_train, y_train)   # LOO on train
X_test  = te_enc.transform(X_test)                  # smoothed mean on test

print("Target-encoded columns (fraud rate per category, smoothed):")
for col in high_card_cols:
    new_col = f"{col}_te"
    print(f"  {new_col}: mean={X_train[new_col].mean():.4f}, "
          f"std={X_train[new_col].std():.4f}")

# Peek at what the encoder learned for P_emaildomain
if "P_emaildomain" in te_enc._mapping:
    print("\\nTop 10 P_emaildomain fraud rates (smoothed):")
    display(te_enc._mapping["P_emaildomain"].sort_values(ascending=False).head(10).to_frame("fraud_rate"))
""")

# ============================================================
# §5 — Interactions
# ============================================================
md("""\
## 6. Hand-Crafted Interaction Features

Domain knowledge about fraud patterns motivates specific feature combinations.
These are features that a model *could* discover, but providing them explicitly
reduces the sample complexity required.

| Feature | Intuition |
|---------|-----------|
| `amt_div_card1_mean_amt` | Is this transaction unusually large for this card? |
| `addr_mismatch` | Purchaser address ≠ recipient address → risky |
| `is_round_amount` | Fraudsters sometimes test with round amounts |
| `log_amt` | Log-transform compresses the heavy right tail |
| `amt_cents` | Cent portion of amount (fraudsters often use .00 or .99) |
""")

code("""\
def add_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # Log-amount (normalises the heavy right tail)
    X["log_TransactionAmt"] = np.log1p(X["TransactionAmt"]).astype(np.float32)

    # Cent portion of amount
    X["amt_cents"] = (X["TransactionAmt"] % 1).round(2).astype(np.float32)

    # Round-amount flag (fraudsters often use $1.00, $10.00, etc.)
    X["is_round_amount"] = (X["amt_cents"] == 0).astype(np.int8)

    # Address mismatch between addr1 and addr2 (if both present and valid)
    if "addr1" in X.columns and "addr2" in X.columns:
        valid = (X["addr1"] != -999) & (X["addr2"] != -999)
        X["addr_mismatch"] = ((X["addr1"] != X["addr2"]) & valid).astype(np.int8)

    # D1 / D2 ratio (relative recency of last transaction)
    if "D1" in X.columns and "D2" in X.columns:
        d2_safe = X["D2"].replace(-999, np.nan)
        d1_safe = X["D1"].replace(-999, np.nan)
        X["D1_D2_ratio"] = (d1_safe / (d2_safe + 1)).astype(np.float32)

    # C1 × C2 interaction (two count features that together may signal fraud)
    if "C1" in X.columns and "C2" in X.columns:
        X["C1_x_C2"] = (X["C1"] * X["C2"]).astype(np.float32)

    return X

X_train = add_interaction_features(X_train)
X_test  = add_interaction_features(X_test)

new_cols = ["log_TransactionAmt", "amt_cents", "is_round_amount",
            "addr_mismatch", "D1_D2_ratio", "C1_x_C2"]
new_cols = [c for c in new_cols if c in X_train.columns]
print(f"Added {len(new_cols)} interaction features.")
display(X_train[new_cols].describe().T)
""")

# ============================================================
# §6 — Aggregations
# ============================================================
md("""\
## 7. Entity-Level Aggregation Features

Fraud patterns are often *entity-level*: a stolen card is used many times
in a short window, or a particular email provider is associated with
unusually high transaction amounts.

Group statistics capture this behaviour:

```
card1_TransactionAmt_mean  = average amount for this card number
card1_TransactionAmt_std   = how volatile is this card's spending?
card1_TransactionAmt_count = how many transactions has this card made?
```

A single transaction that is 5× the card's mean amount is a strong fraud signal
— but the model can only discover this if the group mean is available as a feature.

> **Key best practice:** fit the aggregator on train, transform both train and
> test using stored means. Never let test-set rows contribute to their own group
> statistics.
""")

code("""\
groups = [
    # Card-level behaviour
    {"group_cols": ["card1"],
     "agg_cols": ["TransactionAmt"],
     "agg_funcs": ["mean", "std", "max", "count"]},

    # Card + product interaction
    {"group_cols": ["card1", "ProductCD"],
     "agg_cols": ["TransactionAmt"],
     "agg_funcs": ["mean", "count"]},

    # Email domain risk profiles
    {"group_cols": ["P_emaildomain"],
     "agg_cols": ["TransactionAmt"],
     "agg_funcs": ["mean", "std", "max"]},

    # Address-level patterns
    {"group_cols": ["addr1"],
     "agg_cols": ["TransactionAmt"],
     "agg_funcs": ["mean", "count"]},

    # D-feature (time between transactions) per card
    {"group_cols": ["card1"],
     "agg_cols": ["D1", "D2"],
     "agg_funcs": ["mean", "std"]},
]

agg = GroupAggregator(groups=groups)
X_train = agg.fit_transform(X_train)
X_test  = agg.transform(X_test)

agg_cols = [c for c in X_train.columns if any(
    c.startswith(g["group_cols"][0]) and c not in [g["group_cols"][0]]
    for g in groups
)]
print(f"Aggregation features added. New shape: {X_train.shape}")

# Show a few
agg_feature_names = agg.generated_feature_names
agg_feature_names = [c for c in agg_feature_names if c in X_train.columns]
display(X_train[agg_feature_names[:8]].describe().T)
""")

code("""\
# Visualise: amount vs. card-level mean (fraud vs. legit)
if "card1_TransactionAmt_mean" in X_train.columns:
    plot_df = X_train[["TransactionAmt", "card1_TransactionAmt_mean"]].copy()
    plot_df["ratio"] = plot_df["TransactionAmt"] / (plot_df["card1_TransactionAmt_mean"] + 1)
    plot_df["isFraud"] = y_train.values

    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, ls in [(0, "steelblue", "-"), (1, "tomato", "--")]:
        data = plot_df.loc[plot_df["isFraud"] == label, "ratio"].clip(0, 10)
        sns.kdeplot(data, ax=ax, color=color, ls=ls,
                    label="Legit" if label == 0 else "Fraud", fill=True, alpha=0.3)
    ax.set_xlabel("TransactionAmt / card-mean amount")
    ax.set_title("Distribution of amount-to-card-mean ratio")
    ax.legend()
    plt.tight_layout()
    plt.show()
""")

# ============================================================
# §7 — Temporal
# ============================================================
md("""\
## 8. Temporal Feature Engineering

`TransactionDT` is a *seconds offset* from a fixed reference date
(2017-11-30 in the competition data). By mapping it to a calendar,
we extract actionable signals:

- **Hour of day** — fraud spikes at night (00:00–05:59)
- **Day of week** — weekend spending patterns differ
- **Is-weekend, Is-night** — binary fraud indicators

### Cyclical encoding

Naive linear encoding of `hour` treats 23 and 0 as far apart (distance 23),
but they are only 1 hour apart. Cyclical sin/cos encoding maps the feature
onto a unit circle so that 23:00 and 00:00 are close:

```python
sin_hour = sin(2π × hour / 24)
cos_hour = cos(2π × hour / 24)
```
""")

code("""\
X_train = extract_temporal_features(X_train, col="TransactionDT", cyclical=True)
X_test  = extract_temporal_features(X_test,  col="TransactionDT", cyclical=True)

temporal_cols = ["hour", "day_of_week", "is_weekend", "is_night",
                 "sin_hour", "cos_hour", "tx_age_days"]
temporal_cols = [c for c in temporal_cols if c in X_train.columns]
print(f"Temporal features extracted: {temporal_cols}")
""")

code("""\
# Fraud rate by hour of day
plot_df = pd.DataFrame({"hour": X_train["hour"], "isFraud": y_train.values})
fraud_by_hour = plot_df.groupby("hour")["isFraud"].mean() * 100

fig, ax = plt.subplots(figsize=(11, 4))
fraud_by_hour.plot(kind="bar", ax=ax, color="tomato", edgecolor="white", width=0.8)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud rate by hour of day")
plt.tight_layout()
plt.show()
""")

code("""\
# Email-domain derived features
X_train = extract_email_features(X_train)
X_test  = extract_email_features(X_test)

email_feats = [c for c in X_train.columns
               if "email" in c and c not in ["P_emaildomain", "R_emaildomain"]]
print("Email-derived features:", email_feats)
display(X_train[email_feats].head(5))
""")

# ============================================================
# §8 — PCA on V features
# ============================================================
md("""\
## 9. PCA Compression of V-Feature Blocks

The 339 V-features (V1–V339) were engineered by Vesta Corporation and
grouped into correlated blocks with shared missingness patterns.
Including all 339 raw columns has two problems:

1. **High dimensionality** — many columns are nearly collinear.
2. **Missing-value blocks** — an entire block may be -999 for a given row.

**Strategy:** apply PCA *per block*, retaining 95 % of explained variance.
This typically compresses 50–100 columns to 5–15 components while preserving
the signal.
""")

code("""\
# Define V-feature blocks (mirrors the missingness structure)
V_BLOCKS = {
    "v_blk1":  [f"V{i}" for i in range(1,   12)],
    "v_blk2":  [f"V{i}" for i in range(12,  35)],
    "v_blk3":  [f"V{i}" for i in range(35,  53)],
    "v_blk4":  [f"V{i}" for i in range(53,  75)],
    "v_blk5":  [f"V{i}" for i in range(75,  95)],
    "v_blk6":  [f"V{i}" for i in range(95,  138)],
    "v_blk7":  [f"V{i}" for i in range(138, 167)],
    "v_blk8":  [f"V{i}" for i in range(167, 217)],
    "v_blk9":  [f"V{i}" for i in range(217, 279)],
    "v_blk10": [f"V{i}" for i in range(279, 340)],
}

pca_objects = {}
for block_name, v_cols in V_BLOCKS.items():
    present = [c for c in v_cols if c in X_train.columns]
    if len(present) < 2:
        continue
    X_train, pca_obj = pca_compress(
        X_train, cols=present, n_components=0.95,
        prefix=block_name, fit_df=X_train,
    )
    X_test, _ = pca_compress(
        X_test, cols=present, n_components=pca_obj.n_components_,
        prefix=block_name, fit_df=None,
    )
    # For test we need to apply the already-fitted PCA
    # (pca_compress with int n_components and fit_df=None won't refit)
    pca_objects[block_name] = pca_obj

print(f"\\nShape after PCA compression: {X_train.shape}")
""")

code("""\
# Visualise explained variance for one block
if "v_blk1" in pca_objects:
    plot_pca_explained_variance(pca_objects["v_blk1"],
                                title="PCA — V-block 1 (V1–V11)")
""")

# ============================================================
# §9 — Feature selection
# ============================================================
md("""\
## 10. Feature Selection

With hundreds of engineered features, removing redundant ones:
- Speeds up training
- Reduces overfitting (especially for linear models)
- Improves interpretability

We apply three complementary filters in sequence:

1. **Variance threshold** — drop near-constant features (zero variance).
2. **Correlation pruning** — drop one of each highly-correlated pair (|ρ| > 0.95).
3. **Permutation importance** — keep the top-N most predictive features.
""")

code("""\
# Only keep numeric columns for the selection steps
X_train_num = X_train.select_dtypes(include=[np.number])
X_test_num  = X_test[X_train_num.columns]

print(f"Starting features: {X_train_num.shape[1]}")

# Step 1: Variance threshold
X_train_num = variance_threshold_select(X_train_num, threshold=0.0,
                                        exclude=["isFraud"])
X_test_num  = X_test_num[X_train_num.columns]
print(f"After variance filter: {X_train_num.shape[1]}")

# Step 2: Correlation pruning
print("\\nCorrelation pruning (|ρ| > 0.95):")
X_train_num = drop_high_correlation(X_train_num, threshold=0.95, verbose=True)
X_test_num  = X_test_num[X_train_num.columns]
print(f"After correlation filter: {X_train_num.shape[1]}")
""")

code("""\
# Step 3: Permutation importance with a fast RF
from sklearn.ensemble import RandomForestClassifier

rf_selector = RandomForestClassifier(
    n_estimators=100, max_depth=8, n_jobs=-1, random_state=42
)
rf_selector.fit(X_train_num, y_train)

X_train_selected, imp_series = importance_select(
    X_train_num, y_train, model=rf_selector, top_n=60,
)
X_test_selected = X_test_num[X_train_selected.columns]

print(f"After importance filter (top 60): {X_train_selected.shape[1]}")

plot_feature_importance(
    imp_series, top_n=30,
    title="Permutation importance — top 30 features",
    figsize=(10, 9),
)
""")

# ============================================================
# §10 — Model comparison
# ============================================================
md("""\
## 11. Baseline vs. Engineered Features — Model Lift

We compare two XGBoost classifiers:

- **Baseline** — raw numeric columns only, minimal preprocessing.
- **Engineered** — full feature set built in §§3–10.

This quantifies the value-add of feature engineering beyond raw data.
""")

code("""\
# ── Baseline: raw transaction columns only ────────────────────────────────────
BASELINE_COLS = (
    ["TransactionAmt", "TransactionDT"]
    + [f"card{i}" for i in range(1, 7)]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
)

# Load fresh (un-engineered) splits
X_raw = df.drop(columns=["isFraud", "TransactionID"])
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    X_raw, df["isFraud"], test_size=0.20, random_state=42, stratify=df["isFraud"]
)

base_cols_present = [c for c in BASELINE_COLS
                     if c in X_raw_train.select_dtypes(include=np.number).columns]
X_base_train = X_raw_train[base_cols_present].fillna(-999)
X_base_test  = X_raw_test[base_cols_present].fillna(-999)

print(f"Baseline feature count : {X_base_train.shape[1]}")
print(f"Engineered feature count: {X_train_selected.shape[1]}")
""")

code("""\
xgb_params = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42,
    eval_metric="auc",
    n_jobs=-1,
)

# Baseline model
xgb_base = XGBClassifier(**xgb_params)
xgb_base.fit(X_base_train, y_raw_train)

# Engineered-features model
xgb_eng = XGBClassifier(**xgb_params)
xgb_eng.fit(X_train_selected, y_train)

print("Training complete.")
""")

code("""\
results_base = evaluate_model(
    xgb_base, X_base_train, y_raw_train,
    X_base_test,  y_raw_test,
    name="XGBoost — Baseline features",
)

results_eng = evaluate_model(
    xgb_eng, X_train_selected, y_train,
    X_test_selected, y_test,
    name="XGBoost — Engineered features",
)

lift = results_eng["test_auc"] - results_base["test_auc"]
print(f"\\n>>> AUC lift from feature engineering: {lift:+.4f}")
""")

code("""\
# ROC comparison (handles different feature sets per model)
from sklearn.metrics import roc_curve, auc

fig, ax = plt.subplots(figsize=(8, 6))
pairs = [
    (xgb_base, X_base_test,      y_raw_test, "Baseline",   "steelblue"),
    (xgb_eng,  X_test_selected,  y_test,     "Engineered", "tomato"),
]
for model, X_t, y_t, label, color in pairs:
    proba = model.predict_proba(X_t)[:, 1]
    fpr, tpr, _ = roc_curve(y_t, proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5,
            label=f"{label}  (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — Baseline vs. Feature-Engineered Model")
ax.legend(loc="lower right", fontsize=12)
plt.tight_layout()
plt.show()
""")

# ============================================================
# §11 — Summary
# ============================================================
md("""\
## 12. Summary & Best-Practice Checklist

The table below distils the key lessons from this notebook.

| Practice | Implementation | Why it matters |
|----------|----------------|----------------|
| **Fit-on-train-only** | All encoders, scalers, and aggregators fitted exclusively on `X_train` | Prevents target leakage |
| **Sentinel imputation** | `-999` fill for trees; median + `_was_nan` for linear models | Lets trees learn missingness as a signal |
| **Smoothed target encoding** | `TargetEncoder(smoothing=20)` with LOO | Avoids overfitting on rare categories |
| **Frequency encoding** | `FrequencyEncoder` | Captures category prevalence without leakage |
| **Group aggregations** | `GroupAggregator` | Encodes entity-level behaviour |
| **Cyclical temporals** | `sin/cos` pairs for hour, day-of-week, month | Distances are meaningful for periodic features |
| **PCA per V-block** | `pca_compress` per block | Removes collinearity; respects missingness structure |
| **Correlation pruning** | `drop_high_correlation(threshold=0.95)` | Removes redundancy without harming signal |
| **Permutation importance** | `importance_select(top_n=60)` | Model-agnostic, avoids bias toward high-cardinality |
| **`scale_pos_weight`** | Inverse class-frequency ratio | Handles 3.5 % fraud imbalance |
| **Memory optimisation** | `reduce_mem_usage` | Essential for 590 k × 394 column data |

### Next steps (not covered here)
- **Time-aware cross-validation** — `TimeSeriesSplit` to simulate production conditions.
- **Target-encoded features in CV** — use `cross_val_predict` with target encoder inside each fold.
- **UMAP / t-SNE** — unsupervised exploration of the V-feature latent space.
- **Neural entity embeddings** — learn card and email embeddings end-to-end.
""")

# ============================================================
nb.cells = cells
path = "/home/user/machine_learning_sample/ieee_cis_feature_engineering.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {path}")
