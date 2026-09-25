import time
import os
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
import pickle

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.features.feature_engineering import build_features
from src.evaluation.candidate_recall import evaluate_candidates

def run_task05():
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for row in gt.iter_rows(named=True)}
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    
    # 1. Reproduce Task 04 split to find untouched S1s for calibration
    np.random.seed(42)
    train_s1_subset = set(np.random.choice(train_s1_ids, size=15000, replace=False).tolist())
    train_unseen = [x for x in train_s1_ids if x not in train_s1_subset]
    
    np.random.seed(123) # New seed for calibration
    calib_s1_ids = np.random.choice(train_unseen, size=5000, replace=False).tolist()
    
    print(f"Calibration S1 Set: {len(calib_s1_ids)} (Strictly unseen by T04)")
    
    # Generate features for calibration
    from src.modeling.train_pairwise import generate_train_pairs
    calib_s1_lf = s1.filter(pl.col("entity_id").is_in(calib_s1_ids))
    print("Generating candidates for calibration set...")
    calib_cands = generate_train_pairs(calib_s1_lf, s23, gt_dict, num_s1=5000)
    
    print("Building features for calibration...")
    calib_features = build_features(calib_cands.lazy(), s1, s23)
    feat_cols = [c for c in calib_features.columns if c.startswith("f_")]
    X_calib = calib_features.select(feat_cols).to_numpy()
    y_calib = calib_features["label"].to_numpy()
    
    # Load Model
    with open("src/modeling/lgb_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    print("Scoring calibration set...")
    p_calib_raw = model.predict(X_calib)
    calib_df = calib_features.select(["source1_entity_id", "candidate_entity_id", "label"]).with_columns([
        pl.Series("raw_score", p_calib_raw)
    ])
    
    # Isotonic Calibration
    iso = IsotonicRegression(out_of_bounds='clip')
    p_calib_iso = iso.fit_transform(p_calib_raw, y_calib)
    with open("src/modeling/iso_calibrator.pkl", "wb") as f:
        pickle.dump(iso, f)
        
    calib_df = calib_df.with_columns(pl.Series("iso_score", p_calib_iso))
    
    # Function to compute Macro F0.5
    def compute_macro_f05(preds_df, s1_list):
        pred_dict = {r[0]: set(r[1]) for r in preds_df.group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
        f_scores = []
        for s1_id in s1_list:
            trues = gt_dict.get(s1_id, set())
            preds = pred_dict.get(s1_id, set())
            if not trues and not preds: f_scores.append(1.0)
            elif not trues or not preds: f_scores.append(0.0)
            else:
                tp = len(trues & preds)
                prc = tp / len(preds)
                rcl = tp / len(trues)
                f_scores.append((1.25 * prc * rcl) / (0.25 * prc + rcl) if tp > 0 else 0.0)
        return np.mean(f_scores)
        
    # Global threshold sweep
    print("\n--- Calibration: Threshold Sweep ---")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
    best_f05, best_th = 0, 0
    for th in thresholds:
        f05 = compute_macro_f05(calib_df.filter(pl.col("iso_score") >= th), calib_s1_ids)
        print(f"Threshold {th}: F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05, best_th = f05, th
            
    # Source specific (S2 vs S3)
    s2_ids = set(s23.filter(pl.col("source")=="s2").select("entity_id").collect().to_series().to_list())
    calib_df = calib_df.with_columns(
        pl.col("candidate_entity_id").is_in(list(s2_ids)).alias("is_s2")
    )
    
    # One-parent conflict resolution
    print("\n--- Calibration: 1-Parent Conflict Resolution ---")
    def resolve_1parent(df, th):
        # Only keep candidates above threshold
        df_th = df.filter(pl.col("iso_score") >= th)
        # Sort by score desc, then unique by candidate_entity_id (keeps highest scoring S1 for each candidate)
        resolved = df_th.sort("iso_score", descending=True).unique(subset=["candidate_entity_id"], keep="first")
        return resolved
        
    f05_resolved = compute_macro_f05(resolve_1parent(calib_df, best_th), calib_s1_ids)
    print(f"1-Parent Resolved F0.5 (th={best_th}): {f05_resolved:.4f} vs Unresolved: {best_f05:.4f}")
    
    # Save policy
    use_1parent = (f05_resolved > best_f05)
    final_policy = {"threshold": best_th, "use_1parent": use_1parent, "calibrator": "isotonic"}
    with open("src/modeling/decision_policy.pkl", "wb") as f:
        pickle.dump(final_policy, f)
        
    # Final pristine evaluation
    print("\n--- FINAL PRISTINE EVALUATION ---")
    val_preds = pl.read_csv("output/val_predictions.csv")
    
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(val_s1_ids, size=10000, replace=False).tolist())
    val_unseen = [x for x in val_s1_ids if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(val_unseen, size=20000, replace=False).tolist()
    
    p_val_iso = iso.transform(val_preds["raw_score"].to_numpy())
    val_preds = val_preds.with_columns(pl.Series("iso_score", p_val_iso))
    
    # Apply baseline (th=0.5 raw)
    f05_baseline = compute_macro_f05(val_preds.filter(pl.col("raw_score") >= 0.5), strict_val_ids)
    
    # Apply policy
    val_filtered = val_preds.filter(pl.col("iso_score") >= best_th)
    if use_1parent:
        val_filtered = val_filtered.sort("iso_score", descending=True).unique(subset=["candidate_entity_id"], keep="first")
        
    f05_final = compute_macro_f05(val_filtered, strict_val_ids)
    
    # Singleton vs Non-Singleton behavior
    val_s1_counts = {s1: len(gt_dict.get(s1, set())) for s1 in strict_val_ids}
    singletons = [s for s, c in val_s1_counts.items() if c == 0]
    non_singletons = [s for s, c in val_s1_counts.items() if c > 0]
    
    f05_sing = compute_macro_f05(val_filtered.filter(pl.col("source1_entity_id").is_in(singletons)), singletons)
    f05_non = compute_macro_f05(val_filtered.filter(pl.col("source1_entity_id").is_in(non_singletons)), non_singletons)
    
    print(f"Task 04 Baseline Macro F0.5: {f05_baseline:.4f}")
    print(f"Task 05 Optimal Macro F0.5:  {f05_final:.4f}")
    print(f"Singleton F0.5:              {f05_sing:.4f}")
    print(f"Non-Singleton F0.5:          {f05_non:.4f}")
    
    print("DONE.")

if __name__ == "__main__":
    run_task05()
