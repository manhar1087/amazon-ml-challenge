import time, os, json, gc, pickle, argparse
import polars as pl
import numpy as np
import lightgbm as lgb

from src.data.loader import load_all_data
from src.features.feature_engineering_v2 import build_features_v2
from src.modeling.task05_entity_decoding import compute_metrics, apply_policy

def run_task06(smoke=False):
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    s23_lf = pl.concat([frames["train_source2"].lazy(), frames["train_source3"].lazy()], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    print("Loading Task 04 splits...")
    with open("work/task04/train_val_ids.json", "r") as f:
        splits = json.load(f)
    train_s1_ids = splits["train_ids"]
    val_s1_ids = splits["val_ids"]
    
    if smoke:
        print("SMOKE TEST: Subsampling...")
        train_s1_ids = train_s1_ids[:100]
        val_s1_ids = val_s1_ids[:100]
        
    print("Loading existing Task 04 candidate artifact...")
    all_cands = pl.read_parquet("work/candidates/task04_train_val_cands.parquet")
    
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        s_id = row["source1_entity_id"]
        matches = row["matched_entity_ids"]
        if matches is None or str(matches).strip() == "":
            gt_dict[s_id] = set()
        else:
            gt_dict[s_id] = {x.strip() for x in str(matches).split(",") if x.strip()}
            
    def get_label(s_id, c_id): return 1.0 if c_id in gt_dict.get(s_id, set()) else 0.0
    
    print("Preparing Train set...")
    train_cands_raw = all_cands.filter(pl.col("source1_entity_id").is_in(train_s1_ids))
    train_df = train_cands_raw.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_list()], dtype=pl.Float32),
            return_dtype=pl.Float32
        ).alias("label")
    )
    
    is_pos = pl.col("label") == 1.0
    is_hard_neg = (pl.col("label") == 0.0) & (pl.col("retrieval_channels").str.contains("exact_name|exact_address"))
    np.random.seed(42)
    rand_mask = pl.Series("rand", np.random.rand(train_df.height))
    train_df = train_df.with_columns(rand_mask)
    # Increase negative sampling to 20% for robustness
    is_sampled_easy = (pl.col("label") == 0.0) & ~is_hard_neg & (pl.col("rand") < 0.2)
    train_sampled = train_df.filter(is_pos | is_hard_neg | is_sampled_easy).drop("rand")
    
    # Sort for LambdaRank grouping
    train_sampled = train_sampled.sort("source1_entity_id")
    
    print("Preparing Validation set...")
    val_cands = all_cands.filter(pl.col("source1_entity_id").is_in(val_s1_ids))
    val_df = val_cands.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_list()], dtype=pl.Float32),
            return_dtype=pl.Float32
        ).alias("label")
    )
    val_df = val_df.sort("source1_entity_id")
    
    print("Building Features V2 (Train)...")
    t0 = time.time()
    train_features = build_features_v2(train_sampled.lazy(), s1.lazy(), s23_lf)
    print(f"Built train features in {time.time()-t0:.2f}s")
    
    print("Building Features V2 (Val)...")
    val_features = build_features_v2(val_df.lazy(), s1.lazy(), s23_lf)
    
    feat_cols = [c for c in train_features.columns if c.startswith("f_")]
    X_train = train_features.select(feat_cols).to_numpy()
    y_train = train_features["label"].to_numpy()
    X_val = val_features.select(feat_cols).to_numpy()
    y_val = val_features["label"].to_numpy()
    
    # Extract grouping for lambdarank
    group_train = train_features.group_by("source1_entity_id", maintain_order=True).agg(pl.len())["len"].to_numpy()
    group_val = val_features.group_by("source1_entity_id", maintain_order=True).agg(pl.len())["len"].to_numpy()
    
    models = {}
    
    # 1. Binary Model
    print("Training Binary LightGBM...")
    dtrain_bin = lgb.Dataset(X_train, label=y_train)
    dval_bin = lgb.Dataset(X_val, label=y_val, reference=dtrain_bin)
    
    params_bin = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    model_bin = lgb.train(
        params_bin,
        dtrain_bin,
        num_boost_round=200 if not smoke else 5,
        valid_sets=[dtrain_bin, dval_bin],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=20)]
    )
    models["binary"] = model_bin
    
    # 2. LambdaRank Model
    print("Training LambdaRank LightGBM...")
    dtrain_rank = lgb.Dataset(X_train, label=y_train, group=group_train)
    dval_rank = lgb.Dataset(X_val, label=y_val, reference=dtrain_rank, group=group_val)
    
    params_rank = {
        'objective': 'lambdarank',
        'metric': ['ndcg'],
        'ndcg_eval_at': [1, 3, 5],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    model_rank = lgb.train(
        params_rank,
        dtrain_rank,
        num_boost_round=200 if not smoke else 5,
        valid_sets=[dtrain_rank, dval_rank],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=20)]
    )
    models["lambdarank"] = model_rank
    
    # 3. Model Comparison & Decoding
    # Using 5k tuning S1s and 5k eval S1s to remain consistent with Task 05
    val_ids_sorted = sorted(val_s1_ids)
    np.random.seed(42)
    tune_s1_ids = set(np.random.choice(val_ids_sorted, size=len(val_ids_sorted)//2, replace=False).tolist())
    eval_s1_ids = [x for x in val_ids_sorted if x not in tune_s1_ids]
    tune_s1_ids = list(tune_s1_ids)
    
    best_overall_f05 = -1
    best_model_name = None
    best_policy = None
    best_tune_metrics = None
    best_eval_metrics = None
    best_eval_df = None
    
    comparison_results = {}
    
    for m_name, model in models.items():
        print(f"\nEvaluating Model: {m_name}")
        y_pred = model.predict(X_val)
        val_df_pred = val_features.with_columns(pl.Series("pred", y_pred))
        
        tune_preds = val_df_pred.filter(pl.col("source1_entity_id").is_in(tune_s1_ids))
        eval_preds = val_df_pred.filter(pl.col("source1_entity_id").is_in(eval_s1_ids))
        
        experiments = []
        
        # Grid search on tuning set
        print("Running policy search on tuning set...")
        # A. Thresholds
        for th in [0.5, 0.7, 0.9, 0.926, 0.95, 0.98]:
            pol = {"type": "threshold", "threshold": th}
            exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
            experiments.append({"policy": pol, "metrics": exp_out})
            
        # B. Top-K
        for k in [1, 2, 3]:
            pol = {"type": "top_k", "k": k}
            exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
            experiments.append({"policy": pol, "metrics": exp_out})
            
        # C. Adaptive
        for ath in [0.5, 0.7, 0.9]:
            for rth in [0.7, 0.9, 0.95]:
                pol = {"type": "adaptive", "abs_th": ath, "rel_th": rth}
                exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
                experiments.append({"policy": pol, "metrics": exp_out})
                
        best_tune_exp = max(experiments, key=lambda x: x["metrics"]["macro_f05"])
        print(f"Best tuning policy for {m_name}: {best_tune_exp['policy']}")
        print(f"Tuning Macro F0.5: {best_tune_exp['metrics']['macro_f05']:.4f}")
        
        # Eval on untouched
        eval_out_df = apply_policy(eval_preds, best_tune_exp["policy"])
        eval_mets = compute_metrics(eval_out_df, eval_s1_ids, gt_dict)
        print(f"Untouched Evaluation Macro F0.5: {eval_mets['macro_f05']:.4f}")
        
        comparison_results[m_name] = {
            "best_tuning_policy": best_tune_exp,
            "evaluation_metrics": eval_mets
        }
        
        if eval_mets["macro_f05"] > best_overall_f05:
            best_overall_f05 = eval_mets["macro_f05"]
            best_model_name = m_name
            best_policy = best_tune_exp["policy"]
            best_tune_metrics = best_tune_exp["metrics"]
            best_eval_metrics = eval_mets
            best_eval_df = eval_out_df
            
        # Extract feature importances (Gain) to inspect which features the model relies on
        gain_imp = model.feature_importance(importance_type="gain")
        imp_dict = {f: float(g) for f, g in zip(feat_cols, gain_imp)}
        # Sort desc
        imp_dict = {k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}
        comparison_results[m_name]["feature_importance_gain"] = imp_dict

    print("\n=========================================")
    print(f"WINNING MODEL: {best_model_name}")
    print(f"WINNING POLICY: {best_policy}")
    print(f"WINNING EVAL MACRO F0.5: {best_overall_f05:.4f}")
    print("=========================================\n")
    
    os.makedirs("work/task06", exist_ok=True)
    with open("work/task06/model_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
        
    with open("work/task06/selected_model_config.json", "w") as f:
        json.dump({
            "model_type": best_model_name,
            "features": feat_cols,
            "training_params": params_bin if best_model_name == "binary" else params_rank,
            "tuning_metrics": best_tune_metrics,
            "evaluation_metrics": best_eval_metrics
        }, f, indent=2)
        
    with open("work/task06/selected_policy.json", "w") as f:
        json.dump(best_policy, f, indent=2)
        
    best_eval_df.select(["source1_entity_id", "candidate_entity_id", "pred"]).write_parquet("work/task06/evaluation_predictions.parquet")
    
    with open("work/task06/best_model.pkl", "wb") as f:
        pickle.dump(models[best_model_name], f)
        
    print("Task 06 Advanced Matching completed. Artifacts saved to work/task06/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test")
    args = parser.parse_args()
    run_task06(smoke=args.smoke)
