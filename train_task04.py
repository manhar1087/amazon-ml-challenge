import time, os, json, gc, pickle
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, fbeta_score

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.features.feature_engineering import build_features

from generate_task03_holdout import (
    exact_name_blocking, exact_address_blocking,
    postal_house_blocking, cross_script_name_blocking, abbreviation_blocking,
    run_tfidf_chunked, union_candidates
)

def run_task04():
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    s2_lf = frames["train_source2"].lazy()
    s3_lf = frames["train_source3"].lazy()
    s23_lf = pl.concat([s2_lf, s3_lf], how="diagonal")
    
    gt = frames["train_ground_truth"].collect()
    
    print("Splitting dataset...")
    s1_ids_all = gt["source1_entity_id"].to_list()
    t_split, v_split = get_grouped_split(s1_ids_all)
    
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(v_split, size=10000, replace=False).tolist())
    
    # Train set completely disjoint from the pristine holdout and disjoint from val set
    np.random.seed(202)
    train_s1_subset = np.random.choice(t_split, size=10000, replace=False).tolist()
    
    val_s1_subset = list(t03_tune_ids)
    
    combined_s1_ids = train_s1_subset + val_s1_subset
    s1_combined = s1.filter(pl.col("entity_id").is_in(combined_s1_ids))
    
    artifact_path = "work/candidates/task04_train_val_cands.parquet"
    if os.path.exists(artifact_path):
        print(f"Loading existing candidates from {artifact_path}...")
        all_cands = pl.read_parquet(artifact_path)
    else:
        print("Generating canonical candidates for Train/Val sets...")
        c_name = exact_name_blocking(s1_combined.lazy(), s23_lf)
        c_addr = exact_address_blocking(s1_combined.lazy(), s23_lf)
        c_ph = postal_house_blocking(s1_combined.lazy(), s23_lf)
        c_cs = cross_script_name_blocking(s1_combined.lazy(), s23_lf)
        c_abbr = abbreviation_blocking(s1_combined.lazy(), s23_lf)
        
        print("Loading corpus IDs...")
        corpus_eids = frames["train_source2"].select("entity_id").collect().to_series().to_list() + frames["train_source3"].select("entity_id").collect().to_series().to_list()
        
        c_tfidf_name, _ = run_tfidf_chunked(s1_combined, "latin_name", "word", (1,1), 50, "tfidf_name_k50", corpus_eids)
        c_tfidf_addr, _ = run_tfidf_chunked(s1_combined, "latin_address", "word", (1,2), 20, "tfidf_addr_k20", corpus_eids)
        c_tfidf_char, _ = run_tfidf_chunked(s1_combined, "latin_name", "char_wb", (3,4), 20, "tfidf_char_k20", corpus_eids)
        
        all_cands = union_candidates([c_name, c_addr, c_ph, c_cs, c_abbr, c_tfidf_name, c_tfidf_addr, c_tfidf_char])
        
        os.makedirs("work/candidates", exist_ok=True)
        all_cands.write_parquet(artifact_path)
        print(f"Saved candidates to {artifact_path}")
        
    # Build Ground Truth Dictionary
    gt_dict = {s1: set() for s1 in combined_s1_ids}
    for row in gt.iter_rows(named=True):
        s_id = row["source1_entity_id"]
        if s_id in gt_dict:
            matches = row["matched_entity_ids"]
            if matches is not None and matches != "":
                gt_dict[s_id] = {x.strip() for x in str(matches).split(",") if x.strip()}
                
    # Define Positive Pair Coverage after Blocking
    val_cands = all_cands.filter(pl.col("source1_entity_id").is_in(val_s1_subset))
    pred_dict = {s: set() for s in val_s1_subset}
    for row in val_cands.iter_rows(named=True):
        c_id = row["candidate_entity_id"]
        if c_id is not None:
            pred_dict[row["source1_entity_id"]].add(c_id)
            
    retrieved = 0
    total = 0
    for s in val_s1_subset:
        t_set = gt_dict[s]
        p_set = pred_dict[s]
        retrieved += len(t_set & p_set)
        total += len(t_set)
    coverage = retrieved / total if total > 0 else 0
    print(f"Validation Positive Pair Coverage after Task 03 Blocking: {coverage:.4f}")
    
    # Split into Train / Val
    train_cands_raw = all_cands.filter(pl.col("source1_entity_id").is_in(train_s1_subset))
    
    def get_label(s1, cand): return 1.0 if cand in gt_dict.get(s1, set()) else 0.0
    
    print("Labeling train candidates...")
    train_df = train_cands_raw.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_list()], dtype=pl.Float32),
            return_dtype=pl.Float32
        ).alias("label")
    )
    
    is_pos = pl.col("label") == 1.0
    # Hard negative: high scoring TF-IDF or exact matches, but label 0
    is_hard_neg = (pl.col("label") == 0.0) & (pl.col("retrieval_channels").str.contains("exact_name|exact_address"))
    
    np.random.seed(42)
    rand_mask = pl.Series("rand", np.random.rand(train_df.height))
    train_df = train_df.with_columns(rand_mask)
    is_sampled_easy = (pl.col("label") == 0.0) & ~is_hard_neg & (pl.col("rand") < 0.1)
    
    train_sampled = train_df.filter(is_pos | is_hard_neg | is_sampled_easy).drop("rand")
    
    print(f"Train Positives: {train_sampled.filter(pl.col('label')==1.0).height}")
    print(f"Train Negatives: {train_sampled.filter(pl.col('label')==0.0).height}")
    
    print("Building Training Features...")
    t0 = time.time()
    train_features = build_features(train_sampled.lazy(), s1.lazy(), s23_lf)
    print(f"Training features built in {time.time()-t0:.2f}s")
    
    val_cands_df = val_cands.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_list()], dtype=pl.Float32),
            return_dtype=pl.Float32
        ).alias("label")
    )
    
    print("Building Validation Features...")
    t0 = time.time()
    val_features = build_features(val_cands_df.lazy(), s1.lazy(), s23_lf)
    print(f"Validation features built in {time.time()-t0:.2f}s")
    
    print("Training LightGBM Model...")
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
        'verbose': -1,
        'seed': 42
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=20)]
    )
    
    print("Evaluating Model...")
    y_pred = model.predict(X_val)
    roc = roc_auc_score(y_val, y_pred)
    prc = average_precision_score(y_val, y_pred)
    
    val_features = val_features.with_columns(pl.Series("pred", y_pred))
    
    print(f"Validation ROC-AUC: {roc:.4f}")
    print(f"Validation PR-AUC: {prc:.4f}")
    
    # Calculate downstream entity-level F0.5
    # The one-parent property is invalid, so we do NOT enforce unique subsets.
    # We simply select all pairs where pred >= 0.5
    threshold = 0.5
    final_preds = val_features.filter(pl.col("pred") >= threshold)
    
    pred_res = {s: set() for s in val_s1_subset}
    for row in final_preds.iter_rows(named=True):
        pred_res[row["source1_entity_id"]].add(row["candidate_entity_id"])
        
    f_scores = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for s_id in val_s1_subset:
        trues = gt_dict[s_id]
        preds = pred_res[s_id]
        
        tp = len(trues & preds)
        fp = len(preds - trues)
        fn = len(trues - preds)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        if not trues and not preds: f_scores.append(1.0)
        elif not trues or not preds: f_scores.append(0.0)
        else:
            p = tp / len(preds)
            r = tp / len(trues)
            f_scores.append((1.25 * p * r) / (0.25 * p + r) if tp > 0 else 0.0)
            
    macro_f05 = np.mean(f_scores)
    
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    
    print(f"Pair-level Precision (th={threshold}): {precision:.4f}")
    print(f"Pair-level Recall (th={threshold}): {recall:.4f}")
    print(f"Downstream Entity-level Macro F0.5 (th={threshold}): {macro_f05:.4f}")
    
    # Save artifacts
    os.makedirs("work/task04", exist_ok=True)
    with open("work/task04/lgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    schema = [{"name": c, "type": str(train_features[c].dtype)} for c in feat_cols]
    with open("work/task04/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
        
    val_features.select(["source1_entity_id", "candidate_entity_id", "label", "pred"]).write_parquet("work/task04/val_predictions.parquet")
    
    manifest = {
        "roc_auc": roc,
        "pr_auc": prc,
        "pair_precision": precision,
        "pair_recall": recall,
        "entity_macro_f05": float(macro_f05),
        "train_s1_count": len(train_s1_subset),
        "val_s1_count": len(val_s1_subset),
        "positive_pair_coverage_after_blocking": coverage,
        "negative_sampling": "100% hard (exact), 10% random easy",
        "one_parent_property_enforced": False,
        "one_parent_property_valid": False,
        "timestamp": time.time()
    }
    
    with open("work/task04/experiment_config.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    with open("work/task04/train_val_ids.json", "w") as f:
        json.dump({"train_ids": train_s1_subset, "val_ids": val_s1_subset}, f)
        
    print("Task 04 evaluation complete. Artifacts saved to work/task04/")

if __name__ == '__main__':
    run_task04()
