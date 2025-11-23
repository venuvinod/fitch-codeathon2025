# test.py / full_pipeline.py
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor, Pool

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
sector["revenue_frac"] = sector["revenue_pct"] / 100.0

top_sector = (sector.sort_values(["entity_id", "revenue_pct"], ascending=False)
                    .groupby("entity_id")
                    .first()[["nace_level_2_code", "nace_level_1_code"]]
                    .rename(columns={
                        "nace_level_2_code": "top_nace2",
                        "nace_level_1_code": "top_nace1"
                    }))

entropy = (sector.groupby("entity_id")["revenue_frac"]
                 .apply(lambda p: -(p * np.log(p + 1e-12)).sum())
                 .to_frame("sector_entropy"))

lvl2_count = sector.groupby("entity_id")["nace_level_2_code"].nunique().to_frame("nace2_count")
lvl1_count = sector.groupby("entity_id")["nace_level_1_code"].nunique().to_frame("nace1_count")

sector_feat = top_sector.join([entropy, lvl2_count, lvl1_count]).reset_index()

# -----------------------------
# 3) ENVIRONMENTAL ACTIVITIES FEATURES
# -----------------------------
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

    type_pivot = (acts.pivot_table(
                        index="entity_id",
                        columns="activity_type",
                        values="env_score_adjustment",
                        aggfunc="sum")
                  .add_prefix("env_type_sum_"))

    acts_feat = agg_basic.join(type_pivot).reset_index()
else:
    acts_feat = pd.DataFrame({"entity_id": train["entity_id"].unique()})

# -----------------------------
# 4) SDG FEATURES
# -----------------------------
if len(sdg) > 0:
    sdg["flag"] = 1
    sdg_pivot = (sdg.pivot_table(
                        index="entity_id",
                        columns="sdg_id",
                        values="flag",
                        aggfunc="max",
                        fill_value=0)
                 .add_prefix("sdg_"))

    sdg_count = sdg.groupby("entity_id")["sdg_id"].nunique().to_frame("num_sdgs")
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
# 6) LOG REVENUE + ELEC SECTOR FLAG
# -----------------------------
train_full["log_revenue"] = np.log1p(train_full["revenue"])
test_full["log_revenue"]  = np.log1p(test_full["revenue"])

# Electric / energy-intensive sector proxy (strong for Scope 2)
elec_heavy = {"C", "D", "E"}  # Manufacturing, Utilities, Waste/Water
train_full["elec_sector_flag"] = train_full["top_nace1"].isin(elec_heavy).astype(int)
test_full["elec_sector_flag"]  = test_full["top_nace1"].isin(elec_heavy).astype(int)

train_full["rev_x_elec"] = train_full["log_revenue"] * train_full["elec_sector_flag"]
test_full["rev_x_elec"]  = test_full["log_revenue"]  * test_full["elec_sector_flag"]

# -----------------------------
# 7) LEAKAGE-SAFE TARGET ENCODING
# -----------------------------
def oof_target_encode(train_df, test_df, key_col, target_col, n_splits=5, smoothing=10):
    global_mean = train_df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_te = pd.Series(index=train_df.index, dtype=float)

    for tr_idx, val_idx in kf.split(train_df):
        tr_part = train_df.iloc[tr_idx]
        val_part = train_df.iloc[val_idx]

        stats = tr_part.groupby(key_col)[target_col].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)

        train_te.iloc[val_idx] = val_part[key_col].map(smooth).fillna(global_mean)

    stats_all = train_df.groupby(key_col)[target_col].agg(["mean", "count"])
    smooth_all = (stats_all["mean"] * stats_all["count"] + global_mean * smoothing) / (stats_all["count"] + smoothing)
    test_te = test_df[key_col].map(smooth_all).fillna(global_mean)

    return train_te, test_te


def add_sector_te_features(train_df, test_df, target_col, name_prefix):
    sec_train = sector.merge(train_df[["entity_id", target_col]], on="entity_id", how="inner")
    sec_test  = sector[sector["entity_id"].isin(test_df["entity_id"])].copy()

    sec_train["nace2_te"], sec_test["nace2_te"] = oof_target_encode(
        sec_train, sec_test, "nace_level_2_code", target_col
    )

    sec_train["w_te"] = sec_train["nace2_te"] * (sec_train["revenue_pct"] / 100.0)
    sec_test["w_te"]  = sec_test["nace2_te"]  * (sec_test["revenue_pct"] / 100.0)

    w_train = sec_train.groupby("entity_id")["w_te"].sum().rename(f"{name_prefix}_sector_te")
    w_test  = sec_test.groupby("entity_id")["w_te"].sum().rename(f"{name_prefix}_sector_te")

    train_df = train_df.merge(w_train, on="entity_id", how="left")
    test_df  = test_df.merge(w_test, on="entity_id", how="left")

    fill_val = train_df[target_col].mean()
    train_df[f"{name_prefix}_sector_te"] = train_df[f"{name_prefix}_sector_te"].fillna(fill_val)
    test_df[f"{name_prefix}_sector_te"]  = test_df[f"{name_prefix}_sector_te"].fillna(fill_val)

    return train_df, test_df


def add_activity_te(train_df, test_df, target_col, name_prefix):
    if len(acts) == 0:
        return train_df, test_df

    acts_train = acts.merge(train_df[["entity_id", target_col]], on="entity_id", how="inner")
    acts_test  = acts[acts["entity_id"].isin(test_df["entity_id"])].copy()

    acts_train["act_te"], acts_test["act_te"] = oof_target_encode(
        acts_train, acts_test, "activity_code", target_col
    )

    te_train = acts_train.groupby("entity_id")["act_te"].mean().rename(f"{name_prefix}_activity_te")
    te_test  = acts_test.groupby("entity_id")["act_te"].mean().rename(f"{name_prefix}_activity_te")

    train_df = train_df.merge(te_train, on="entity_id", how="left")
    test_df  = test_df.merge(te_test, on="entity_id", how="left")

    fill_val = train_df[target_col].mean()
    train_df[f"{name_prefix}_activity_te"] = train_df[f"{name_prefix}_activity_te"].fillna(fill_val)
    test_df[f"{name_prefix}_activity_te"]  = test_df[f"{name_prefix}_activity_te"].fillna(fill_val)

    return train_df, test_df


train_full, test_full = add_sector_te_features(train_full, test_full, "target_scope_1", "scope1")
train_full, test_full = add_sector_te_features(train_full, test_full, "target_scope_2", "scope2")

train_full, test_full = add_activity_te(train_full, test_full, "target_scope_1", "scope1")
train_full, test_full = add_activity_te(train_full, test_full, "target_scope_2", "scope2")

# -----------------------------
# 8) FINAL FEATURE CLEANUP
# -----------------------------
all_env = pd.concat([train_full["env_adj_sum"], test_full["env_adj_sum"]], axis=0)
env_min, env_max = all_env.min(), all_env.max()
if env_max > env_min:
    train_full["env_adj_sum_norm"] = (train_full["env_adj_sum"] - env_min) / (env_max - env_min)
    test_full["env_adj_sum_norm"]  = (test_full["env_adj_sum"] - env_min) / (env_max - env_min)
else:
    train_full["env_adj_sum_norm"] = 0
    test_full["env_adj_sum_norm"]  = 0

train_full["rev_x_scope1_sector_te"] = train_full["log_revenue"] * train_full["scope1_sector_te"]
train_full["rev_x_scope2_sector_te"] = train_full["log_revenue"] * train_full["scope2_sector_te"]
test_full["rev_x_scope1_sector_te"]  = test_full["log_revenue"]  * test_full["scope1_sector_te"]
test_full["rev_x_scope2_sector_te"]  = test_full["log_revenue"]  * test_full["scope2_sector_te"]

# -----------------------------
# 9) TARGETS (LOG SPACE) + SAMPLE WEIGHTS
# -----------------------------
y1_raw = train_full["target_scope_1"].values
y2_raw = train_full["target_scope_2"].values

target1 = np.log1p(y1_raw)
target2 = np.log1p(y2_raw)

# gentle emphasis on large emitters
w1 = np.log1p(y1_raw) + 1.0
w2 = np.log1p(y2_raw) + 1.0

# -----------------------------
# 10) FEATURES + CATEGORICALS
# -----------------------------
drop_cols = ["target_scope_1", "target_scope_2", "entity_id"]
feature_cols = [c for c in train_full.columns if c not in drop_cols]

X_train = train_full[feature_cols].copy()
X_test  = test_full[feature_cols].copy()

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

for c in cat_cols:
    X_train[c] = X_train[c].astype(str)
    X_test[c]  = X_test[c].astype(str)

cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

# -----------------------------
# 11) TRAIN MODELS (LOG + WEIGHTS)
# -----------------------------
def train_catboost(X, y_log, cat_features, label_name, y_raw, weights, params):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_log = np.zeros(len(X))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_log[tr_idx], y_log[val_idx]
        w_tr, w_val = weights[tr_idx], weights[val_idx]

        train_pool = Pool(X_tr, y_tr, cat_features=cat_features, weight=w_tr)
        val_pool   = Pool(X_val, y_val, cat_features=cat_features, weight=w_val)

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        preds_val_log = model.predict(val_pool)
        oof_log[val_idx] = preds_val_log
        models.append(model)

        rmse_log = np.sqrt(mean_squared_error(y_val, preds_val_log))
        mae_log  = mean_absolute_error(y_val, preds_val_log)
        r2_log   = r2_score(y_val, preds_val_log)

        y_val_raw = y_raw[val_idx]
        preds_val_raw = np.expm1(preds_val_log)

        rmse_raw = np.sqrt(mean_squared_error(y_val_raw, preds_val_raw))
        mae_raw  = mean_absolute_error(y_val_raw, preds_val_raw)
        r2_raw   = r2_score(y_val_raw, preds_val_raw)

        print(
            f"[{label_name}] Fold {fold+1} | "
            f"RMSE_log: {rmse_log:.4f} | MAE_log: {mae_log:.4f} | R²_log: {r2_log:.4f} || "
            f"RMSE_raw: {rmse_raw:,.2f} | MAE_raw: {mae_raw:,.2f} | R²_raw: {r2_raw:.4f}"
        )

    overall_rmse_log = np.sqrt(mean_squared_error(y_log, oof_log))
    overall_mae_log  = mean_absolute_error(y_log, oof_log)
    overall_r2_log   = r2_score(y_log, oof_log)

    oof_raw = np.expm1(oof_log)
    overall_rmse_raw = np.sqrt(mean_squared_error(y_raw, oof_raw))
    overall_mae_raw  = mean_absolute_error(y_raw, oof_raw)
    overall_r2_raw   = r2_score(y_raw, oof_raw)

    print("\n==============================")
    print(f"[{label_name}] OVERALL OOF (LOG)")
    print(f"RMSE_log: {overall_rmse_log:.4f} | MAE_log: {overall_mae_log:.4f} | R²_log: {overall_r2_log:.4f}")
    print(f"[{label_name}] OVERALL OOF (RAW)")
    print(f"RMSE_raw: {overall_rmse_raw:,.2f} | MAE_raw: {overall_mae_raw:,.2f} | R²_raw: {overall_r2_raw:.4f}")
    print("==============================\n")

    return models


params_scope1 = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=250,
    verbose=200
)

params_scope2 = dict(
    iterations=6000,
    learning_rate=0.025,
    depth=10,
    l2_leaf_reg=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=300,
    verbose=200
)

models1 = train_catboost(X_train, target1, cat_idx, "Scope 1", y1_raw, w1, params_scope1)
models2 = train_catboost(X_train, target2, cat_idx, "Scope 2", y2_raw, w2, params_scope2)

# -----------------------------
# 12) PREDICT TEST
# -----------------------------
def predict_ensemble(models, X, cat_features):
    pool = Pool(X, cat_features=cat_features)
    preds = np.mean([m.predict(pool) for m in models], axis=0)
    return preds

pred1_log = predict_ensemble(models1, X_test, cat_idx)
pred2_log = predict_ensemble(models2, X_test, cat_idx)

pred1 = np.expm1(pred1_log)
pred2 = np.expm1(pred2_log)

# -----------------------------
# 13) WRITE SUBMISSION
# -----------------------------
sub = pd.DataFrame({
    "entity_id": test_full["entity_id"],
    "target_scope_1": pred1,
    "target_scope_2": pred2
})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv ✅")
