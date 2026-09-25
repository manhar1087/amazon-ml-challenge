import polars as pl
import numpy as np
from typing import Dict

def evaluate_candidates(candidates_df: pl.DataFrame, y_true: dict, s1_val_ids: list):
    """
    Evaluates candidate generation recall and volume metrics.
    
    candidates_df: DataFrame with columns [source1_entity_id, candidate_entity_id]
    y_true: dict of ground truth {s1_id: set([s2_ids])}
    s1_val_ids: list of validation S1 IDs
    """
    
    val_s1_set = set(s1_val_ids)
    
    # Filter ground truth for validation
    val_true = {k: set(v) for k, v in y_true.items() if k in val_s1_set}
    
    # Filter candidates for validation
    cands_val = candidates_df.filter(pl.col("source1_entity_id").is_in(list(val_s1_set)))
    
    # Create dict of predictions
    grouped = cands_val.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
    pred_dict = {row[0]: set(row[1]) for row in grouped.iter_rows()}
    
    # Add empty sets for S1s with no candidates
    for s1 in val_s1_set:
        if s1 not in pred_dict:
            pred_dict[s1] = set()
            
    # Calculate metrics
    total_true_matched_ids = 0
    total_true_retrieved = 0
    
    macro_recall_scores = []
    full_set_recovered = 0
    non_singleton_count = 0
    
    candidate_counts = []
    
    for s1_id in val_s1_set:
        true_set = val_true.get(s1_id, set())
        pred_set = pred_dict.get(s1_id, set())
        
        c_count = len(pred_set)
        candidate_counts.append(c_count)
        
        if len(true_set) == 0:
            pass
        else:
            intersection = true_set & pred_set
            retrieved_count = len(intersection)
            true_count = len(true_set)
            
            total_true_matched_ids += true_count
            total_true_retrieved += retrieved_count
            
            recall = retrieved_count / true_count
            macro_recall_scores.append(recall)
            
            non_singleton_count += 1
            if retrieved_count == true_count:
                full_set_recovered += 1
                
    pair_level_recall = total_true_retrieved / total_true_matched_ids if total_true_matched_ids > 0 else 0.0
    macro_recall = sum(macro_recall_scores) / len(macro_recall_scores) if macro_recall_scores else 0.0
    full_set_recovery_rate = full_set_recovered / non_singleton_count if non_singleton_count > 0 else 0.0
    
    counts_arr = np.array(candidate_counts)
    
    return {
        "pair_level_recall": pair_level_recall,
        "macro_candidate_recall": macro_recall,
        "full_set_recovery_rate": full_set_recovery_rate,
        "avg_candidates_per_s1": float(np.mean(counts_arr)) if len(counts_arr) else 0.0,
        "median_candidates": float(np.median(counts_arr)) if len(counts_arr) else 0.0,
        "p95_candidates": float(np.percentile(counts_arr, 95)) if len(counts_arr) else 0.0,
        "p99_candidates": float(np.percentile(counts_arr, 99)) if len(counts_arr) else 0.0,
        "max_candidates": float(np.max(counts_arr)) if len(counts_arr) else 0.0,
        "total_candidates": int(np.sum(counts_arr)) if len(counts_arr) else 0
    }

def print_recall_report(metrics: dict, name: str = "Candidate Recall Report"):
    print("="*40)
    print(f"       {name}")
    print("="*40)
    print(f"Pair-level Recall:         {metrics['pair_level_recall']:.5f}")
    print(f"Macro Candidate Recall:    {metrics['macro_candidate_recall']:.5f}")
    print(f"Full-Set Recovery Rate:    {metrics['full_set_recovery_rate']:.5f}")
    print(f"Avg Candidates / S1:       {metrics['avg_candidates_per_s1']:.2f}")
    print(f"Median Candidates / S1:    {metrics['median_candidates']:.2f}")
    print(f"95th %ile Candidates:      {metrics['p95_candidates']:.2f}")
    print(f"99th %ile Candidates:      {metrics['p99_candidates']:.2f}")
    print(f"Max Candidates:            {metrics['max_candidates']}")
    print(f"Total Candidate Pairs:     {metrics['total_candidates']}")
    print("="*40)
