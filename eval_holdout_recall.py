import polars as pl
import json
import numpy as np

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split

def is_indic(text):
    if not text: return False
    # simple heuristic: check for Devanagari or other Indic unicode blocks
    return any('\u0900' <= c <= '\u0d7f' for c in str(text))

def evaluate_artifact():
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    gt = frames["train_ground_truth"].collect()
    
    t_split, v_split = get_grouped_split(gt['source1_entity_id'].to_list())
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(v_split, size=10000, replace=False).tolist())
    v_unseen = [x for x in v_split if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(v_unseen, size=20000, replace=False).tolist()
    
    s1_20k = s1.filter(pl.col("entity_id").is_in(strict_val_ids))
    s1_dict = {row["entity_id"]: row for row in s1_20k.iter_rows(named=True)}
    
    print("Loading candidate artifact...")
    cands = pl.read_parquet("work/candidates/task03_frozen_holdout_20k.parquet").drop_nulls()
    
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for row in gt.iter_rows(named=True)}
    
    c_dict = {r[0]: set(r[1]) for r in cands.group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
    
    def get_stats(subset_ids):
        t_true = 0
        t_found = 0
        t_missed = 0
        macro_sum = 0.0
        full_sets = 0
        s1_count = 0
        for sid in subset_ids:
            t = gt_dict.get(sid, set())
            c = c_dict.get(sid, set())
            if not t: continue
            
            s1_count += 1
            t_true += len(t)
            found = len(t & c)
            t_found += found
            t_missed += (len(t) - found)
            
            macro_sum += found / len(t)
            if found == len(t):
                full_sets += 1
                
        pair_rec = t_found / t_true if t_true > 0 else 0
        macro_rec = macro_sum / s1_count if s1_count > 0 else 0
        full_set_rec = full_sets / s1_count if s1_count > 0 else 0
        return t_true, t_found, t_missed, pair_rec, macro_rec, full_set_rec
        
    print("\n--- PART 2: Candidate recall evaluation ---")
    
    # Base Stats
    tt, tf, tm, tr, mr, fs = get_stats(strict_val_ids)
    print(f"Total true matched IDs: {tt}")
    print(f"True matched IDs present in candidates: {tf}")
    print(f"True matched IDs missed by blocking: {tm}")
    print(f"Pair candidate recall: {tr:.4f}")
    print(f"Macro candidate recall: {mr:.4f}")
    print(f"Full-set recovery: {fs:.4f}")
    
    cand_counts = [len(c_dict.get(s, set())) for s in strict_val_ids]
    print(f"Total candidate pairs: {cands.height}")
    print(f"Average candidates/S1: {np.mean(cand_counts):.2f}")
    print(f"Median candidates/S1: {np.median(cand_counts):.2f}")
    print(f"P95: {np.percentile(cand_counts, 95):.2f}")
    print(f"P99: {np.percentile(cand_counts, 99):.2f}")
    print(f"Max: {np.max(cand_counts)}")
    
    # Breakdowns
    print("\n--- Breakdown ---")
    us_ids = [s for s in strict_val_ids if s1_dict[s]["country"] == "US"]
    in_ids = [s for s in strict_val_ids if s1_dict[s]["country"] == "IN"]
    indic_ids = [s for s in strict_val_ids if is_indic(s1_dict[s]["business_name"])]
    mix_ids = [s for s in strict_val_ids if is_indic(s1_dict[s]["business_name"]) and any('a'<=c.lower()<='z' for c in str(s1_dict[s]["business_name"]))]
    miss_addr_ids = [s for s in strict_val_ids if not s1_dict[s]["business_address"]]
    
    print(f"US: {get_stats(us_ids)[3]:.4f}")
    print(f"India: {get_stats(in_ids)[3]:.4f}")
    print(f"Indic-containing: {get_stats(indic_ids)[3]:.4f}")
    print(f"Mixed-script: {get_stats(mix_ids)[3]:.4f}")
    print(f"Missing-address: {get_stats(miss_addr_ids)[3]:.4f}")
    
    # S2/S3 breakdown
    s2_t, s2_f = 0, 0
    s3_t, s3_f = 0, 0
    for sid in strict_val_ids:
        t = gt_dict.get(sid, set())
        c = c_dict.get(sid, set())
        for m in t:
            if m.startswith("S2"):
                s2_t += 1
                if m in c: s2_f += 1
            else:
                s3_t += 1
                if m in c: s3_f += 1
                
    print(f"S2: {s2_f/s2_t if s2_t > 0 else 0:.4f}")
    print(f"S3: {s3_f/s3_t if s3_t > 0 else 0:.4f}")
    
    # Channel stats (Part 3)
    print("\n--- PART 3: Channel Incremental Volume ---")
    # For this we need to re-group by channel, but we collapsed channels in union_candidates!
    # Wait, the artifact only has a combined "retrieval_channels" column.
    
if __name__ == "__main__":
    evaluate_artifact()
