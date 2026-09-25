import time
import os
import polars as pl
import functools
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
import pickle

print = functools.partial(print, flush=True)

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.features.feature_engineering import build_features
from src.evaluation.candidate_recall import evaluate_candidates

def run_verification():
    print("Loading data for Verification Audit...")
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for row in gt.iter_rows(named=True)}
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    
    # 1. Validation Independence Setup
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(val_s1_ids, size=10000, replace=False).tolist())
    
    # Strict unseen holdout
    val_unseen = [x for x in val_s1_ids if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(val_unseen, size=20000, replace=False).tolist()
    strict_val_set = set(strict_val_ids)
    
    np.random.seed(42)
    train_s1_subset = np.random.choice(train_s1_ids, size=15000, replace=False).tolist()
    
    print(f"Task 03 Tune Set: {len(t03_tune_ids)}")
    print(f"Task 04 Train Set: {len(train_s1_subset)}")
    print(f"Task 04 Strict Unseen Val Set: {len(strict_val_ids)}")
    
    print("\nLoading precomputed validation candidates...")
    val_cands = pl.read_csv("output/candidate_pairs.tsv", separator="\t")
    val_cands = val_cands.with_columns(pl.col("candidate_entity_ids").str.split(",")).explode("candidate_entity_ids").rename({"candidate_entity_ids": "candidate_entity_id"}).filter(pl.col("candidate_entity_id") != "")
    
    # Filter to strict val
    val_cands = val_cands.filter(pl.col("source1_entity_id").is_in(strict_val_ids))
    
    def get_label(s1, cand):
        return 1.0 if cand in gt_dict.get(s1, set()) else 0.0
        
    val_cands = val_cands.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_dicts()]),
            return_dtype=pl.Float32
        ).alias("label")
    )
    
    # 2. Positive Coverage
    total_val_gt_pairs = sum(len(gt_dict.get(s1, set())) for s1 in strict_val_ids)
    val_cands_collected = val_cands.collect()
    available_positives = val_cands_collected.filter(pl.col("label") == 1.0).height
    missed_by_blocking = total_val_gt_pairs - available_positives
    
    print(f"\nTotal True Positives in Holdout GT: {total_val_gt_pairs}")
    print(f"Positive Candidates Available to Classifier: {available_positives}")
    print(f"Missed purely by Blocking (Task 03 Loss): {missed_by_blocking}")
    
    # 3. Training Negative Distribution
    # To save time, we will dynamically sample train pairs exactly as we did before.
    print("\nRegenerating Training Candidates for exact metrics...")
    from src.modeling.train_pairwise import generate_train_pairs
    train_s1_lf = s1.filter(pl.col("entity_id").is_in(train_s1_subset))
    train_cands_sampled = generate_train_pairs(train_s1_lf, s23, gt_dict, num_s1=15000)
    
    t_pos = train_cands_sampled.filter(pl.col("label")==1.0).height
    t_tot = train_cands_sampled.height
    print(f"Exact Train Positives: {t_pos}")
    print(f"Exact Train Total Pairs: {t_tot}")
    print(f"Exact Train Positive Rate: {t_pos/t_tot:.4f}")
    
    print("\nBuilding Features...")
    val_features = build_features(val_cands_collected.lazy(), s1, s23)
    train_features = build_features(train_cands_sampled.lazy(), s1, s23)
    
    feat_cols = [c for c in train_features.columns if c.startswith("f_")]
    X_train = train_features.select(feat_cols).to_numpy()
    y_train = train_features["label"].to_numpy()
    X_val = val_features.select(feat_cols).to_numpy()
    y_val = val_features["label"].to_numpy()
    
    print("\nTraining Baseline...")
    params = {'objective': 'binary', 'metric': ['auc', 'average_precision'], 'learning_rate': 0.1, 'num_leaves': 31, 'verbose': -1}
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    m_base = lgb.train(params, dtrain, num_boost_round=100, valid_sets=[dtrain, dval], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(False)])
    
    def score_model(model, name):
        p = model.predict(X_val)
        pr = average_precision_score(y_val, p)
        roc = roc_auc_score(y_val, p)
        b = (p > 0.5).astype(int)
        prec = precision_score(y_val, b)
        rec = recall_score(y_val, b)
        
        # Macro F0.5
        v_df = val_features.with_columns(pl.Series("p", p))
        pred_dict = {r[0]: set(r[1]) for r in v_df.filter(pl.col("p")>0.5).group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
        f_scores = []
        fp_total = 0
        for s1_id in strict_val_ids:
            trues = gt_dict.get(s1_id, set())
            preds = pred_dict.get(s1_id, set())
            fp_total += len(preds - trues)
            if not trues and not preds: f_scores.append(1.0)
            elif not trues or not preds: f_scores.append(0.0)
            else:
                tp = len(trues & preds)
                prc = tp / len(preds)
                rcl = tp / len(trues)
                f_scores.append((1.25 * prc * rcl) / (0.25 * prc + rcl) if tp > 0 else 0.0)
        
        print(f"\nModel: {name}")
        print(f"ROC-AUC: {roc:.4f}, PR-AUC: {pr:.4f}")
        print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")
        print(f"Macro F0.5: {np.mean(f_scores):.4f}")
        print(f"Avg FP per S1: {fp_total / len(strict_val_ids):.4f}")
        return p
        
    p_base = score_model(m_base, "Baseline")
    
    print("\n4. HNM Execution...")
    p_tr = m_base.predict(X_train)
    fp_mask = (y_train == 0)
    fp_scores = p_tr[fp_mask]
    top_fp_idx = np.argsort(fp_scores)[-8000:]
    w = np.ones(len(y_train))
    w[np.where(fp_mask)[0][top_fp_idx]] = 5.0
    print(f"Upweighted {len(top_fp_idx)} Hard Negatives from TRAIN only.")
    
    dtrain_hnm = lgb.Dataset(X_train, label=y_train, weight=w)
    m_hnm = lgb.train(params, dtrain_hnm, num_boost_round=100, valid_sets=[dtrain_hnm, dval], callbacks=[lgb.early_stopping(10), lgb.log_evaluation(False)])
    p_hnm = score_model(m_hnm, "HNM Model")
    
    print("\n8. Saving Validation Scores...")
    out_df = val_features.select(["source1_entity_id", "candidate_entity_id"]).with_columns([
        pl.Series("raw_score", p_hnm)
    ])
    os.makedirs("output", exist_ok=True)
    out_df.write_csv("output/val_predictions.csv")
    print("Saved to output/val_predictions.csv")

if __name__ == "__main__":
    run_verification()
