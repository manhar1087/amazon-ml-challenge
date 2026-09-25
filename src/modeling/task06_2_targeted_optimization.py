import time, os, json, gc, pickle, argparse
import polars as pl
import numpy as np
import lightgbm as lgb

from src.data.loader import load_all_data
from src.features.feature_engineering_v2 import build_features_v2
from src.features.feature_engineering_v3 import build_features_v3
from src.modeling.task05_entity_decoding import compute_metrics, apply_policy

def run_task06_2(smoke=False):
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
        train_s1_ids = train_s1_ids[:20]
        val_s1_ids = val_s1_ids[:20]
        
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
    is_sampled_easy = (pl.col("label") == 0.0) & ~is_hard_neg & (pl.col("rand") < 0.2)
    train_sampled = train_df.filter(is_pos | is_hard_neg | is_sampled_easy).drop("rand").sort("source1_entity_id")
    
    print("Preparing Validation set...")
    val_cands = all_cands.filter(pl.col("source1_entity_id").is_in(val_s1_ids))
    val_df = val_cands.with_columns(
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([get_label(r["source1_entity_id"], r["candidate_entity_id"]) for r in s.to_list()], dtype=pl.Float32),
            return_dtype=pl.Float32
        ).alias("label")
    ).sort("source1_entity_id")
    
    print("Building Features V2...")
    t_v2_train = build_features_v2(train_sampled.lazy(), s1.lazy(), s23_lf)
    t_v2_val = build_features_v2(val_df.lazy(), s1.lazy(), s23_lf)
    
    print("Building Features V3...")
    t_v3_train = build_features_v3(train_sampled.lazy(), s1.lazy(), s23_lf)
    t_v3_val = build_features_v3(val_df.lazy(), s1.lazy(), s23_lf)
    
    params_bin = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42
    }
    
    models = {}
    feature_sets = {
        "Binary_V2": (t_v2_train, t_v2_val),
        "Binary_V3": (t_v3_train, t_v3_val)
    }
    
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
    
    comparison_results = {}
    
    for m_name, (df_train, df_val) in feature_sets.items():
        print(f"\nTraining Model: {m_name}")
        feat_cols = [c for c in df_train.columns if c.startswith("f_")]
        
        X_train = df_train.select(feat_cols).to_numpy()
        y_train = df_train["label"].to_numpy()
        X_val = df_val.select(feat_cols).to_numpy()
        y_val = df_val["label"].to_numpy()
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params_bin,
            dtrain,
            num_boost_round=300 if not smoke else 5,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=20)]
        )
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        y_pred = model.predict(X_val)
        roc_auc = roc_auc_score(y_val, y_pred) if len(np.unique(y_val)) > 1 else 0.0
        pr_auc = average_precision_score(y_val, y_pred) if sum(y_val) > 0 else 0.0
        
        val_df_pred = df_val.with_columns(pl.Series("pred", y_pred))
        tune_preds = val_df_pred.filter(pl.col("source1_entity_id").is_in(tune_s1_ids))
        eval_preds = val_df_pred.filter(pl.col("source1_entity_id").is_in(eval_s1_ids))
        
        experiments = []
        for th in [0.5, 0.7, 0.9, 0.926, 0.95]:
            pol = {"type": "threshold", "threshold": th}
            exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
            experiments.append({"policy": pol, "metrics": exp_out})
            
        for k in [1, 2, 3]:
            pol = {"type": "top_k", "k": k}
            exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
            experiments.append({"policy": pol, "metrics": exp_out})
            
        for ath in [0.5, 0.7, 0.9]:
            for rth in [0.7, 0.9, 0.95]:
                pol = {"type": "adaptive", "abs_th": ath, "rel_th": rth}
                exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
                experiments.append({"policy": pol, "metrics": exp_out})
                
        best_tune_exp = max(experiments, key=lambda x: x["metrics"]["macro_f05"])
        eval_out_df = apply_policy(eval_preds, best_tune_exp["policy"])
        eval_mets = compute_metrics(eval_out_df, eval_s1_ids, gt_dict)
        
        gain_imp = model.feature_importance(importance_type="gain")
        imp_dict = {f: float(g) for f, g in zip(feat_cols, gain_imp)}
        imp_dict = {k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}
        
        comparison_results[m_name] = {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_tuning_policy": best_tune_exp,
            "evaluation_metrics": eval_mets,
            "feature_importance_gain": imp_dict
        }
        
        print(f"[{m_name}] ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
        print(f"[{m_name}] Untouched Eval Macro F0.5: {eval_mets['macro_f05']:.4f}")
        
        if eval_mets["macro_f05"] > best_overall_f05:
            best_overall_f05 = eval_mets["macro_f05"]
            best_model_name = m_name
            best_policy = best_tune_exp["policy"]
            best_tune_metrics = best_tune_exp["metrics"]
            best_eval_metrics = eval_mets

    print("\n=========================================")
    print(f"WINNING MODEL: {best_model_name}")
    print(f"WINNING POLICY: {best_policy}")
    print(f"WINNING EVAL MACRO F0.5: {best_overall_f05:.4f}")
    print("=========================================\n")
    
    os.makedirs("work/task06", exist_ok=True)
    with open("work/task06/model_comparison_v3.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_task06_2(smoke=args.smoke)
