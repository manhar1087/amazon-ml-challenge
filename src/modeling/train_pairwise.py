import time
import os
import polars as pl
import functools
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, fbeta_score
import pickle

print = functools.partial(print, flush=True)

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.features.feature_engineering import build_features, get_feature_groups
from src.evaluation.candidate_recall import evaluate_candidates

# Candidate generation imports for training
from src.blocking.union import union_candidates
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from src.blocking.cross_script import cross_script_name_blocking
from src.blocking.abbreviation import abbreviation_blocking
from src.blocking.sparse_retrieval import sparse_top_k_retrieval

def generate_train_pairs(s1_lf, s23_lf, gt_dict, num_s1=10000):
    # Generates candidates for a small training slice to fit in memory
    print(f"Generating training candidates for {num_s1} S1 entities...")
    s1_ids_all = s1_lf.select("entity_id").collect().to_series().to_list()
    np.random.seed(42)
    s1_subset = np.random.choice(s1_ids_all, size=num_s1, replace=False).tolist()
    s1_val_lf = s1_lf.filter(pl.col("entity_id").is_in(s1_subset))
    s23_lf = s23_lf.filter(pl.col("latin_name") != "")
    
    b_name = exact_name_blocking(s1_val_lf, s23_lf)
    b_addr = exact_address_blocking(s1_val_lf, s23_lf)
    b_ph = postal_house_blocking(s1_val_lf, s23_lf)
    
    # We will just use these fast blocks for training pairs, plus a small sparse for hard negatives
    b_cs = cross_script_name_blocking(s1_val_lf, s23_lf)
    b_abbr = abbreviation_blocking(s1_val_lf, s23_lf)
    
    # Tiny sparse for training hard negatives
    s1_val_lf_sm = s1_val_lf.head(num_s1)
    b_word = sparse_top_k_retrieval(s1_val_lf_sm, s23_lf, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=20, channel_name="tfidf_name", chunk_size=10000, max_df=0.01)
    
    cands = union_candidates([b_name, b_addr, b_ph, b_cs, b_abbr, b_word])
    
    # Labeling
    def get_label(s1, cand):
        return 1.0 if cand in gt_dict.get(s1, set()) else 0.0
        
    cands_df = cands.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_dicts()]),
            return_dtype=pl.Float32
        ).alias("label")
    )
    
    print(f"Generated {cands_df.height} raw training candidates.")
    
    # Negative Sampling Strategy:
    # - 100% of positives
    # - 100% of hard negatives (e.g. exact name match but label 0)
    # - 10% of easy negatives
    is_pos = pl.col("label") == 1.0
    is_hard_neg = (pl.col("label") == 0.0) & (pl.col("retrieval_channels").str.contains("exact_name|exact_address"))
    
    np.random.seed(42)
    rand_mask = pl.Series("rand", np.random.rand(cands_df.height))
    cands_df = cands_df.with_columns(rand_mask)
    is_sampled_easy = (pl.col("label") == 0.0) & ~is_hard_neg & (pl.col("rand") < 0.1)
    
    sampled_df = cands_df.filter(is_pos | is_hard_neg | is_sampled_easy).drop("rand")
    
    pos_count = sampled_df.filter(pl.col("label") == 1.0).height
    neg_count = sampled_df.filter(pl.col("label") == 0.0).height
    print(f"Training Positives: {pos_count}, Negatives: {neg_count} (Ratio: 1:{neg_count/max(1,pos_count):.2f})")
    
    return sampled_df

def run_pairwise():
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for row in gt.iter_rows(named=True)}
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    
    train_s1_lf = s1.filter(pl.col("entity_id").is_in(train_s1_ids))
    val_s1_lf = s1.filter(pl.col("entity_id").is_in(val_s1_ids))
    
    train_cands_df = generate_train_pairs(train_s1_lf, s23, gt_dict, num_s1=10000)
    
    print("\nBuilding Training Features...")
    t0 = time.time()
    train_features = build_features(train_cands_df.lazy(), train_s1_lf, s23)
    print(f"Training features built in {time.time()-t0:.2f}s")
    
    print("\nLoading Validation Candidates (from output/candidate_pairs.tsv)...")
    val_cands = pl.read_csv("output/candidate_pairs.tsv", separator="\t")
    # explode comma separated
    val_cands = val_cands.with_columns(pl.col("candidate_entity_ids").str.split(",")).explode("candidate_entity_ids").rename({"candidate_entity_ids": "candidate_entity_id"}).filter(pl.col("candidate_entity_id") != "")
    
    # We will sample the validation candidates heavily so feature engineering completes quickly
    # A 5% random subset of val_s1_ids (approx 20,000 S1s)
    np.random.seed(99)
    val_subset = np.random.choice(val_s1_ids, size=20000, replace=False).tolist()
    val_cands = val_cands.filter(pl.col("source1_entity_id").is_in(val_subset))
    
    def get_val_label(s1, cand):
        return 1.0 if cand in gt_dict.get(s1, set()) else 0.0
    val_cands = val_cands.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_val_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_dicts()]),
            return_dtype=pl.Float32
        ).alias("label")
    )
    
    print("\nBuilding Validation Features...")
    val_features = build_features(val_cands.lazy(), val_s1_lf, s23)
    
    print("\nTraining LightGBM Model...")
    feat_cols = [c for c in train_features.columns if c.startswith("f_")]
    X_train = train_features.select(feat_cols).to_numpy()
    y_train = train_features["label"].to_numpy()
    
    X_val = val_features.select(feat_cols).to_numpy()
    y_val = val_features["label"].to_numpy()
    
    params = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=100,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=20)]
    )
    
    os.makedirs("src/modeling", exist_ok=True)
    with open("src/modeling/lgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print("\nEvaluating Baseline Model on Validation...")
    val_preds = model.predict(X_val)
    pr_auc = average_precision_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_preds)
    
    # Trivial threshold 0.5
    preds_binary = (val_preds > 0.5).astype(int)
    prec = precision_score(y_val, preds_binary)
    rec = recall_score(y_val, preds_binary)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"Precision (thr=0.5): {prec:.4f}")
    print(f"Recall (thr=0.5):    {rec:.4f}")
    
    # Feature Ablation (Group evaluation)
    print("\nFeature Ablation Study (PR-AUC on Val):")
    groups = get_feature_groups()
    base_features = groups["name"].copy()
    
    # 1. Name only
    X_tr_abl = train_features.select(base_features).to_numpy()
    X_va_abl = val_features.select(base_features).to_numpy()
    m_name = lgb.train(params, lgb.Dataset(X_tr_abl, label=y_train), num_boost_round=50, verbose_eval=False)
    p_name = m_name.predict(X_va_abl)
    print(f"Name only: {average_precision_score(y_val, p_name):.4f}")
    
    # 2. + Address
    base_features += groups["address"]
    X_tr_abl = train_features.select(base_features).to_numpy()
    X_va_abl = val_features.select(base_features).to_numpy()
    m_addr = lgb.train(params, lgb.Dataset(X_tr_abl, label=y_train), num_boost_round=50, verbose_eval=False)
    p_addr = m_addr.predict(X_va_abl)
    print(f"+ Address: {average_precision_score(y_val, p_addr):.4f}")
    
    # 3. + Cross + Context
    base_features += groups["cross"] + groups["context"]
    X_tr_abl = train_features.select(base_features).to_numpy()
    X_va_abl = val_features.select(base_features).to_numpy()
    m_cross = lgb.train(params, lgb.Dataset(X_tr_abl, label=y_train), num_boost_round=50, verbose_eval=False)
    p_cross = m_cross.predict(X_va_abl)
    print(f"+ Cross/Ctx: {average_precision_score(y_val, p_cross):.4f}")
    
    print("\nHard Negative Mining Round 1...")
    # Find false positives from training predictions
    train_preds = model.predict(X_train)
    # Get indices of top 5000 False Positives
    fp_mask = (y_train == 0)
    fp_scores = train_preds[fp_mask]
    if len(fp_scores) > 0:
        top_fp_idx = np.argsort(fp_scores)[-5000:]
        
        # We can simulate HNM by boosting the weight of these hard negatives, 
        # or dynamically querying more. Here we simply upweight them 5x.
        weights = np.ones(len(y_train))
        fp_indices_in_orig = np.where(fp_mask)[0][top_fp_idx]
        weights[fp_indices_in_orig] = 5.0
        
        dtrain_hnm = lgb.Dataset(X_train, label=y_train, weight=weights)
        model_hnm = lgb.train(
            params,
            dtrain_hnm,
            num_boost_round=100,
            valid_sets=[dtrain_hnm, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=False)]
        )
        
        val_preds_hnm = model_hnm.predict(X_val)
        pr_auc_hnm = average_precision_score(y_val, val_preds_hnm)
        print(f"HNM Round 1 PR-AUC: {pr_auc_hnm:.4f} (Delta: {pr_auc_hnm - pr_auc:+.4f})")
    
    print("\nPreliminary S1 Macro F0.5 Diagnostic (Threshold=0.5)")
    val_scored = val_features.with_columns(pl.Series("pred", val_preds))
    pred_matches = val_scored.filter(pl.col("pred") > 0.5)
    
    # Calculate Macro F0.5
    s1_groups = pred_matches.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
    pred_dict = {row[0]: set(row[1]) for row in s1_groups.iter_rows()}
    
    f_scores = []
    for s1_id in val_subset:
        trues = gt_dict.get(s1_id, set())
        preds = pred_dict.get(s1_id, set())
        
        if len(trues) == 0 and len(preds) == 0:
            f_scores.append(1.0)
        elif len(trues) == 0 and len(preds) > 0:
            f_scores.append(0.0)
        elif len(trues) > 0 and len(preds) == 0:
            f_scores.append(0.0)
        else:
            tp = len(trues & preds)
            precision = tp / len(preds) if len(preds) > 0 else 0
            recall = tp / len(trues) if len(trues) > 0 else 0
            if tp == 0:
                f_scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                f_scores.append(f05)
                
    macro_f05 = np.mean(f_scores)
    print(f"Macro F0.5 (thr=0.5): {macro_f05:.4f}")
    
    print("\nDONE.")

if __name__ == "__main__":
    run_pairwise()
