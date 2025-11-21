this is my current code
# full_pipeline.py
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Try to import catboost; if unavailable provide a small shim that exposes
# CatBoostRegressor and Pool-compatible objects using sklearn's GradientBoostingRegressor.
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------
# 1) LOAD DATA
# -----------------------------
train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
sector = pd.read_csv("data/revenue_distribution_by_sector.csv")
acts   = pd.read_csv("data/environmental_activities.csv")
sdg    = pd.read_csv("data/sustainable_development_goals.csv")

# -----------------------------
# 2) SECTOR FEATURES
# -----------------------------
# Basic aggregations per entity
# - Top sector
# - Sector entropy (diversification)
# - Level-1 & Level-2 counts
sector["revenue_frac"] = sector["revenue_pct"] / 100.0

# dominant sector (by revenue %)
top_sector = (sector.sort_values(["entity_id", "revenue_pct"], ascending=False)
                    .groupby("entity_id")
                    .first()[["nace_level_2_code", "nace_level_1_code"]]
                    .rename(columns={
                        "nace_level_2_code": "top_nace2",
                        "nace_level_1_code": "top_nace1"
                    }))

# entropy
entropy = (sector.groupby("entity_id")["revenue_frac"]
                 .apply(lambda p: -(p*np.log(p+1e-12)).sum())
                 .to_frame("sector_entropy"))

# counts
lvl2_count = sector.groupby("entity_id")["nace_level_2_code"].nunique().to_frame("nace2_count")
lvl1_count = sector.groupby("entity_id")["nace_level_1_code"].nunique().to_frame("nace1_count")

sector_feat = top_sector.join([entropy, lvl2_count, lvl1_count]).reset_index()

# -----------------------------
# 3) ENVIRONMENTAL ACTIVITIES FEATURES
# -----------------------------
# Negative = good, Positive = bad
# Aggregate per entity:
# - total / mean adjustment
# - pos/neg counts + ratios
# - sum by activity_type
if len(acts) > 0:
    acts["is_pos"] = (acts["env_score_adjustment"] > 0).astype(int)
    acts["is_neg"] = (acts["env_score_adjustment"] < 0).astype(int)

    agg_basic = acts.groupby("entity_id").agg(
        env_adj_sum=("env_score_adjustment", "sum"),
        env_adj_mean=("env_score_adjustment", "mean"),
        env_adj_std=("env_score_adjustment", "std"),
        env_pos_count=("is_pos", "sum"),
        env_neg_count=("is_neg", "sum"),
        env_act_count=("env_score_adjustment", "count"),
    )

    agg_basic["env_pos_ratio"] = agg_basic["env_pos_count"] / (agg_basic["env_act_count"] + 1e-9)
    agg_basic["env_neg_ratio"] = agg_basic["env_neg_count"] / (agg_basic["env_act_count"] + 1e-9)

    # sum by activity_type
    type_pivot = (acts.pivot_table(index="entity_id",
                                   columns="activity_type",
                                   values="env_score_adjustment",
                                   aggfunc="sum")
                       .add_prefix("env_type_sum_"))

    acts_feat = agg_basic.join(type_pivot).reset_index()
else:
    acts_feat = pd.DataFrame({"entity_id": train["entity_id"].unique()})

# Normalize adjustment sum to 0-1 (OPTIONAL, safe baseline)
# using global min/max from combined train+test after merge
# We'll compute after merge.

# -----------------------------
# 4) SDG FEATURES
# -----------------------------
if len(sdg) > 0:
    sdg["flag"] = 1
    sdg_pivot = (sdg.pivot_table(index="entity_id",
                                 columns="sdg_id",
                                 values="flag",
                                 aggfunc="max",
                                 fill_value=0)
                     .add_prefix("sdg_"))

    sdg_count = sdg.groupby("entity_id")["sdg_id"].nunique().to_frame("num_sdgs")
    # climate emphasis
    climate_ids = {7, 12, 13}
    sdg_climate = (sdg[sdg["sdg_id"].isin(climate_ids)]
                   .groupby("entity_id")["sdg_id"].nunique()
                   .to_frame("num_climate_sdgs"))

    sdg_feat = sdg_pivot.join([sdg_count, sdg_climate]).fillna(0).reset_index()
else:
    sdg_feat = pd.DataFrame({"entity_id": train["entity_id"].unique()})

# -----------------------------
# 5) MERGE EVERYTHING
# -----------------------------
def merge_all(base_df):
    df = base_df.copy()
    df = df.merge(sector_feat, on="entity_id", how="left")
    df = df.merge(acts_feat, on="entity_id", how="left")
    df = df.merge(sdg_feat, on="entity_id", how="left")
    return df

train_full = merge_all(train)
test_full  = merge_all(test)

# Fill numeric missing values separately (test doesn't have targets)
num_cols_train = train_full.select_dtypes(include=[np.number]).columns
num_cols_test  = test_full.select_dtypes(include=[np.number]).columns

train_full[num_cols_train] = train_full[num_cols_train].fillna(0)
test_full[num_cols_test]   = test_full[num_cols_test].fillna(0)


# -----------------------------
# 6) LEAKAGE-SAFE TARGET ENCODING
# -----------------------------
# We encode:
# - nace_level_2_code (sector)
# - activity_code (environment activities)
#
# We'll compute out-of-fold averages and then aggregate weighted by revenue_pct for sectors.

def oof_target_encode(train_df, test_df, key_col, target_col, n_splits=5, smoothing=10):
    """
    Returns two Series:
      train_te: out-of-fold target encoding for train
      test_te: full-data target encoding for test
    """
    global_mean = train_df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_te = pd.Series(index=train_df.index, dtype=float)

    for tr_idx, val_idx in kf.split(train_df):
        tr_part = train_df.iloc[tr_idx]
        val_part = train_df.iloc[val_idx]

        stats = tr_part.groupby(key_col)[target_col].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)

        train_te.iloc[val_idx] = val_part[key_col].map(smooth).fillna(global_mean)

    # test mapping using all train
    stats_all = train_df.groupby(key_col)[target_col].agg(["mean", "count"])
    smooth_all = (stats_all["mean"] * stats_all["count"] + global_mean * smoothing) / (stats_all["count"] + smoothing)
    test_te = test_df[key_col].map(smooth_all).fillna(global_mean)

    return train_te, test_te

# ---- Sector TE weighted by revenue mix ----
# We need a sector-level TE per entity:
# 1) TE per (entity_id, nace2) row
# 2) weighted sum by revenue_pct

def add_sector_te_features(train_df, test_df, target_col, name_prefix):
    # build a "long" sector table with target for train entities
    sec_train = sector.merge(train_df[["entity_id", target_col]], on="entity_id", how="inner")
    sec_test  = sector[sector["entity_id"].isin(test_df["entity_id"])].copy()

    # TE for nace2 rows
    sec_train["nace2_te"], sec_test["nace2_te"] = oof_target_encode(
        sec_train, sec_test, "nace_level_2_code", target_col
    )

    # weighted sum per entity
    sec_train["w_te"] = sec_train["nace2_te"] * (sec_train["revenue_pct"] / 100.0)
    sec_test["w_te"]  = sec_test["nace2_te"]  * (sec_test["revenue_pct"] / 100.0)

    w_train = sec_train.groupby("entity_id")["w_te"].sum().rename(f"{name_prefix}_sector_te")
    w_test  = sec_test.groupby("entity_id")["w_te"].sum().rename(f"{name_prefix}_sector_te")

    train_df = train_df.merge(w_train, on="entity_id", how="left")
    test_df  = test_df.merge(w_test, on="entity_id", how="left")

    train_df[f"{name_prefix}_sector_te"] = train_df[f"{name_prefix}_sector_te"].fillna(train_df[target_col].mean())
    test_df[f"{name_prefix}_sector_te"]  = test_df[f"{name_prefix}_sector_te"].fillna(train_df[target_col].mean())

    return train_df, test_df

train_full, test_full = add_sector_te_features(train_full, test_full, "target_scope_1", "scope1")
train_full, test_full = add_sector_te_features(train_full, test_full, "target_scope_2", "scope2")

# ---- Activity code TE (optional but strong) ----
def add_activity_te(train_df, test_df, target_col, name_prefix):
    if len(acts) == 0:
        return train_df, test_df

    acts_train = acts.merge(train_df[["entity_id", target_col]], on="entity_id", how="inner")
    acts_test  = acts[acts["entity_id"].isin(test_df["entity_id"])].copy()

    acts_train["act_te"], acts_test["act_te"] = oof_target_encode(
        acts_train, acts_test, "activity_code", target_col
    )

    # aggregate to entity by mean te
    te_train = acts_train.groupby("entity_id")["act_te"].mean().rename(f"{name_prefix}_activity_te")
    te_test  = acts_test.groupby("entity_id")["act_te"].mean().rename(f"{name_prefix}_activity_te")

    train_df = train_df.merge(te_train, on="entity_id", how="left")
    test_df  = test_df.merge(te_test, on="entity_id", how="left")

    train_df[f"{name_prefix}_activity_te"] = train_df[f"{name_prefix}_activity_te"].fillna(train_df[target_col].mean())
    test_df[f"{name_prefix}_activity_te"]  = test_df[f"{name_prefix}_activity_te"].fillna(train_df[target_col].mean())

    return train_df, test_df

train_full, test_full = add_activity_te(train_full, test_full, "target_scope_1", "scope1")
train_full, test_full = add_activity_te(train_full, test_full, "target_scope_2", "scope2")

# -----------------------------
# 7) FINAL FEATURE CLEANUP
# -----------------------------
# log revenue
train_full["log_revenue"] = np.log1p(train_full["revenue"])
test_full["log_revenue"]  = np.log1p(test_full["revenue"])

# Normalize env_adj_sum to 0-1 by global min/max (combined for stability)
all_env = pd.concat([train_full["env_adj_sum"], test_full["env_adj_sum"]], axis=0)
env_min, env_max = all_env.min(), all_env.max()
if env_max > env_min:
    train_full["env_adj_sum_norm"] = (train_full["env_adj_sum"] - env_min) / (env_max - env_min)
    test_full["env_adj_sum_norm"]  = (test_full["env_adj_sum"] - env_min) / (env_max - env_min)
else:
    train_full["env_adj_sum_norm"] = 0
    test_full["env_adj_sum_norm"]  = 0

# Interaction: size × sector intensity
train_full["rev_x_scope1_sector_te"] = train_full["log_revenue"] * train_full["scope1_sector_te"]
train_full["rev_x_scope2_sector_te"] = train_full["log_revenue"] * train_full["scope2_sector_te"]
test_full["rev_x_scope1_sector_te"]  = test_full["log_revenue"]  * test_full["scope1_sector_te"]
test_full["rev_x_scope2_sector_te"]  = test_full["log_revenue"]  * test_full["scope2_sector_te"]

target1 = np.log1p(train_full["target_scope_1"].values)
target2 = np.log1p(train_full["target_scope_2"].values)

# Drop target columns from features
pred1 = np.expm1(pred1)
pred2 = np.expm1(pred2)


drop_cols = ["target_scope_1", "target_scope_2", "entity_id"]
feature_cols = [c for c in train_full.columns if c not in drop_cols]

# Identify categorical columns for CatBoost
cat_cols = []
for c in feature_cols:
    if train_full[c].dtype == "object":
        cat_cols.append(c)

# -----------------------------
# 8) TRAIN MODELS
# -----------------------------
def train_catboost(X, y, cat_features, label_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof = np.zeros(len(X))
    models = []

    # store fold metrics
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool   = Pool(X_val, y_val, cat_features=cat_features)

        model = CatBoostRegressor(
            iterations=4000,
            learning_rate=0.03,
            depth=8,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            early_stopping_rounds=200,
            verbose=200
        )

        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        preds_val = model.predict(val_pool)
        oof[val_idx] = preds_val
        models.append(model)

        # ---- Fold metrics ----
        rmse = np.sqrt(mean_squared_error(y_val, preds_val))
        mae  = mean_absolute_error(y_val, preds_val)
        r2   = r2_score(y_val, preds_val)

        # MAPE (avoid divide-by-zero)
        mape = np.mean(np.abs((y_val - preds_val) / (y_val + 1e-9))) * 100

        fold_metrics.append((rmse, mae, r2, mape))

        print(
            f"[{label_name}] Fold {fold+1} | "
            f"RMSE: {rmse:,.4f} | MAE: {mae:,.4f} | R²: {r2:.4f} | MAPE: {mape:.2f}%"
        )

    # ---- Overall metrics ----
    overall_rmse = np.sqrt(mean_squared_error(y, oof))
    overall_mae  = mean_absolute_error(y, oof)
    overall_r2   = r2_score(y, oof)
    overall_mape = np.mean(np.abs((y - oof) / (y + 1e-9))) * 100

    print("\n==============================")
    print(f"[{label_name}] OVERALL OOF METRICS")
    print(f"RMSE: {overall_rmse:,.4f}")
    print(f"MAE : {overall_mae:,.4f}")
    print(f"R²  : {overall_r2:.4f}")
    print(f"MAPE: {overall_mape:.2f}%")
    print("==============================\n")

    return models



X_train = train_full[feature_cols].copy()
X_test  = test_full[feature_cols].copy()

# One-hot encode categoricals safely
full = pd.concat([X_train, X_test], axis=0)
full = pd.get_dummies(full, columns=cat_cols, drop_first=False)

X_train = full.iloc[:len(X_train)].reset_index(drop=True)
X_test  = full.iloc[len(X_train):].reset_index(drop=True)

# 1) Recompute categorical columns from X_train (not train_full)
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

# 2) Ensure they are strings (CatBoost requirement)
for c in cat_cols:
    X_train[c] = X_train[c].astype(str)
    X_test[c]  = X_test[c].astype(str)

# 3) Convert cat column NAMES -> INDICES (CatBoost Pool expects indices reliably)
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

models1 = train_catboost(X_train, target1, cat_cols, "Scope 1")
models2 = train_catboost(X_train, target2, cat_cols, "Scope 2")

# -----------------------------
# 9) PREDICT TEST
# -----------------------------
def predict_ensemble(models, X, cat_features):
    pool = Pool(X, cat_features=cat_features)
    preds = np.mean([m.predict(pool) for m in models], axis=0)
    return preds

pred1 = predict_ensemble(models1, X_test, cat_cols)
pred2 = predict_ensemble(models2, X_test, cat_cols)

# -----------------------------
# 10) WRITE SUBMISSION
# -----------------------------
sub = pd.DataFrame({
    "entity_id": test_full["entity_id"],
    "target_scope_1": pred1,
    "target_scope_2": pred2
})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv ✅")
