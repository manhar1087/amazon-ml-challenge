import json
import os
import argparse
import time
import numpy as np
import polars as pl
from typing import Dict, List, Set, Any

def compute_metrics(preds_df: pl.DataFrame, s1_ids: List[str], gt_dict: Dict[str, Set[str]]) -> Dict[str, Any]:
    """
    Computes Macro F0.5 and cardinality metrics per S1.
    preds_df must have: source1_entity_id, candidate_entity_id (only the predicted pairs)
    """
    # Build prediction dict
    pred_dict = {s1: set() for s1 in s1_ids}
    if preds_df.height > 0:
        # Avoid groupby if empty
        grouped = preds_df.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
        for row in grouped.iter_rows():
            s1_id = row[0]
            if s1_id in pred_dict:
                pred_dict[s1_id] = set(row[1])

    f_scores = []
    zero_match_f05 = []
    singleton_f05 = []
    multi_match_f05 = []
    
    under_predicted = 0
    over_predicted = 0
    exact_predicted = 0
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for s1_id in s1_ids:
        T = gt_dict.get(s1_id, set())
        P = pred_dict.get(s1_id, set())
        
        len_T = len(T)
        len_P = len(P)
        
        tp = len(T & P)
        fp = len(P - T)
        fn = len(T - P)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        if len_P < len_T: under_predicted += 1
        elif len_P > len_T: over_predicted += 1
        else: exact_predicted += 1
        
        # Challenge Logic
        if len_T == 0 and len_P == 0:
            score = 1.0
        elif len_T == 0 and len_P > 0:
            score = 0.0
        elif len_T > 0 and len_P == 0:
            score = 0.0
        else:
            precision = tp / len_P
            recall = tp / len_T
            if tp > 0:
                score = (1.25 * precision * recall) / (0.25 * precision + recall)
            else:
                score = 0.0
                
        f_scores.append(score)
        if len_T == 0:
            zero_match_f05.append(score)
        elif len_T == 1:
            singleton_f05.append(score)
        else:
            multi_match_f05.append(score)
            
    return {
        "macro_f05": float(np.mean(f_scores)) if f_scores else 0.0,
        "zero_match_f05": float(np.mean(zero_match_f05)) if zero_match_f05 else 0.0,
        "singleton_f05": float(np.mean(singleton_f05)) if singleton_f05 else 0.0,
        "multi_match_f05": float(np.mean(multi_match_f05)) if multi_match_f05 else 0.0,
        "exact_predicted_count": exact_predicted,
        "under_predicted_count": under_predicted,
        "over_predicted_count": over_predicted,
        "total_predicted_pairs": sum(len(pred_dict[s1]) for s1 in s1_ids),
        "total_TP": tp_total,
        "total_FP": fp_total,
        "total_FN": fn_total
    }

def apply_policy(df: pl.DataFrame, policy: Dict[str, Any]) -> pl.DataFrame:
    """
    Applies a decoding policy to the raw prediction DataFrame.
    Expects df with: source1_entity_id, candidate_entity_id, pred
    """
    ptype = policy.get("type", "threshold")
    
    # Pre-sort by prediction score descending to facilitate top-K and adaptive rules
    df_sorted = df.sort(
        ["source1_entity_id", "pred", "candidate_entity_id"],
        descending=[False, True, False]
    )
    
    if ptype == "threshold":
        th = policy.get("threshold", 0.5)
        return df_sorted.filter(pl.col("pred") >= th)
        
    elif ptype == "top_k":
        k = policy.get("k", 1)
        if k == 0:
            return df_sorted.filter(pl.lit(False))
        return df_sorted.group_by("source1_entity_id", maintain_order=True).head(k)
        
    elif ptype == "threshold_top_k":
        th = policy.get("threshold", 0.5)
        k = policy.get("k", 1)
        if k == 0:
            return df_sorted.filter(pl.lit(False))
        filtered = df_sorted.filter(pl.col("pred") >= th)
        return filtered.group_by("source1_entity_id", maintain_order=True).head(k)
        
    elif ptype == "adaptive":
        # Keep if pred >= absolute_th AND pred >= relative_th * top1_pred
        abs_th = policy.get("abs_th", 0.5)
        rel_th = policy.get("rel_th", 0.0)
        
        filtered = df_sorted.filter(pl.col("pred") >= abs_th)
        if rel_th > 0:
            top1 = filtered.group_by("source1_entity_id").agg(pl.col("pred").max().alias("max_pred"))
            filtered = filtered.join(top1, on="source1_entity_id")
            filtered = filtered.filter(pl.col("pred") >= pl.col("max_pred") * rel_th)
            filtered = filtered.drop("max_pred")
            
        k = policy.get("k", None)
        if k is not None:
            filtered = filtered.group_by("source1_entity_id", maintain_order=True).head(k)
            
        return filtered
        
    else:
        raise ValueError(f"Unknown policy type: {ptype}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test on 50 S1s")
    args = parser.parse_args()

    print("Loading Task 04 validation IDs...")
    with open("work/task04/train_val_ids.json", "r") as f:
        splits = json.load(f)
    
    val_ids = splits.get("val_ids", [])
    if not val_ids:
        raise ValueError("val_ids not found in train_val_ids.json")
        
    print(f"Total Validation S1s from Task 04: {len(val_ids)}")
    
    # Deterministic split 5k tuning, 5k eval
    val_ids_sorted = sorted(val_ids) # Sort for determinism
    np.random.seed(42)
    tune_s1_ids = set(np.random.choice(val_ids_sorted, size=len(val_ids_sorted)//2, replace=False).tolist())
    eval_s1_ids = [x for x in val_ids_sorted if x not in tune_s1_ids]
    tune_s1_ids = list(tune_s1_ids)
    
    if args.smoke:
        print("SMOKE TEST MODE: Using 50 S1s for tuning, 50 for eval")
        tune_s1_ids = tune_s1_ids[:50]
        eval_s1_ids = eval_s1_ids[:50]
        
    print(f"Tuning S1s: {len(tune_s1_ids)}")
    print(f"Evaluation S1s: {len(eval_s1_ids)}")
    
    print("Loading ground truth...")
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    gt_dict = {}
    
    for row in gt_df.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        matches = row["matched_entity_ids"]
        if matches is None or str(matches).strip() == "":
            gt_dict[s1] = set()
        else:
            gt_dict[s1] = {x.strip() for x in str(matches).split(",") if x.strip()}
            
    print("Loading Task 04 predictions...")
    preds_df = pl.read_parquet("work/task04/val_predictions.parquet")
    
    # Make sure we only use the specified S1s
    tune_preds = preds_df.filter(pl.col("source1_entity_id").is_in(tune_s1_ids))
    eval_preds = preds_df.filter(pl.col("source1_entity_id").is_in(eval_s1_ids))
    
    experiments = []
    
    # A. Global threshold baseline
    print("\nA. Evaluating Global Threshold Baseline...")
    policy_base = {"type": "threshold", "threshold": 0.926}
    exp_out = compute_metrics(apply_policy(tune_preds, policy_base), tune_s1_ids, gt_dict)
    experiments.append({"name": "Baseline Th=0.926", "policy": policy_base, "metrics": exp_out})
    
    # B. Fixed Top-K
    print("\nB. Evaluating Fixed Top-K...")
    for k in range(0, 12):
        pol = {"type": "top_k", "k": k}
        exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
        experiments.append({"name": f"Fixed Top-{k}", "policy": pol, "metrics": exp_out})
        
    # C. Threshold + Top-K
    print("\nC. Evaluating Threshold + Top-K Grid...")
    thresholds = [0.5, 0.7, 0.9, 0.926, 0.95, 0.98]
    ks = [1, 2, 3, 5, 10]
    for th in thresholds:
        for k in ks:
            pol = {"type": "threshold_top_k", "threshold": th, "k": k}
            exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
            experiments.append({"name": f"Th={th} Top-{k}", "policy": pol, "metrics": exp_out})
            
    # D. Adaptive Cardinality
    print("\nD. Evaluating Adaptive Cardinality...")
    abs_ths = [0.5, 0.7, 0.9, 0.926]
    rel_ths = [0.5, 0.7, 0.8, 0.9, 0.95]
    for ath in abs_ths:
        for rth in rel_ths:
            pol = {"type": "adaptive", "abs_th": ath, "rel_th": rth}
            exp_out = compute_metrics(apply_policy(tune_preds, pol), tune_s1_ids, gt_dict)
            experiments.append({"name": f"Adaptive abs={ath} rel={rth}", "policy": pol, "metrics": exp_out})
            
    # Find best policy on tuning set
    best_exp = max(experiments, key=lambda x: x["metrics"]["macro_f05"])
    print("\n=========================================")
    print("BEST POLICY ON TUNING SET:")
    print(json.dumps(best_exp, indent=2))
    print("=========================================\n")
    
    # Apply to evaluation set
    print("Applying best policy to Untouched Evaluation Set...")
    eval_df = apply_policy(eval_preds, best_exp["policy"])
    eval_metrics = compute_metrics(eval_df, eval_s1_ids, gt_dict)
    
    final_result = {
        "tuning_best_experiment": best_exp,
        "evaluation_metrics": eval_metrics
    }
    
    print("FINAL EVALUATION METRICS:")
    print(json.dumps(eval_metrics, indent=2))
    
    os.makedirs("work/task05", exist_ok=True)
    
    with open("work/task05/decoding_experiment_results.json", "w") as f:
        json.dump(experiments, f, indent=2)
        
    with open("work/task05/selected_policy.json", "w") as f:
        json.dump(final_result, f, indent=2)
        
    # Save final evaluation predictions
    eval_df.select(["source1_entity_id", "candidate_entity_id", "pred"]).write_parquet("work/task05/evaluation_predictions.parquet")
    print("\nSaved artifacts to work/task05/")

if __name__ == "__main__":
    main()
