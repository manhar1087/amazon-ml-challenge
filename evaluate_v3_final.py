import time
import os
import polars as pl
import functools
import numpy as np

print = functools.partial(print, flush=True)

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.evaluation.candidate_recall import evaluate_candidates, print_recall_report
from src.blocking.union import union_candidates
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from src.blocking.cross_script import cross_script_name_blocking
from src.blocking.sparse_retrieval import sparse_top_k_retrieval
from src.blocking.abbreviation import abbreviation_blocking

def run_v3_rescue():
    print("Loading data for Rescue Analysis...")
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    
    # 10k subset
    np.random.seed(101)
    val_subset_ids = np.random.choice(val_s1_ids, size=10000, replace=False).tolist()
    val_subset_set = set(val_subset_ids)
    
    s1_val_lf = s1.filter(pl.col("entity_id").is_in(val_subset_ids))
    
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        if row["source1_entity_id"] in val_subset_set:
            matches = row["matched_entity_ids"]
            gt_dict[row["source1_entity_id"]] = set(matches.split(",")) if matches else set()
            
    print("\n[1] Computing Base Channels...")
    b_name = exact_name_blocking(s1_val_lf, s23)
    b_addr = exact_address_blocking(s1_val_lf, s23)
    b_ph = postal_house_blocking(s1_val_lf, s23)
    b_cs = cross_script_name_blocking(s1_val_lf, s23)
    b_abbr = abbreviation_blocking(s1_val_lf, s23)
    
    print("\n[2] Computing Top-100 TF-IDF Channels...")
    t0 = time.time()
    # K=100 for all
    b_word = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=100, channel_name="tfidf_name_word", max_df=0.01)
    b_addr_tf = sparse_top_k_retrieval(s1_val_lf, s23, col_name="normalized_address", analyzer="word", ngram_range=(1,2), k=100, channel_name="tfidf_address_word", max_df=0.02)
    b_char = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="char_wb", ngram_range=(3,4), k=100, channel_name="tfidf_name_char", max_df=0.005)
    print(f"Sparse Extraction completed in {time.time()-t0:.2f}s")
    
    # Helper to slice K
    def slice_k(df, k):
        return df.group_by("source1_entity_id").head(k)
        
    print("\n[3] Executing Final Union K-Sweep...")
    
    sweep_results = []
    baseline_true_retrieved = 0
    k_vals = [20, 30, 50, 75, 100]
    
    final_cands_100 = None
    metrics_100 = None
    
    for k in k_vals:
        print(f"\n--- Union K={k} ---")
        w_k = slice_k(b_word, k)
        a_k = slice_k(b_addr_tf, k)
        c_k = slice_k(b_char, k)
        
        cands = union_candidates([b_name, b_addr, b_ph, b_cs, b_abbr, w_k, a_k, c_k])
        metrics = evaluate_candidates(cands, gt_dict, val_subset_ids)
        print_recall_report(metrics, f"FINAL UNION (K={k})")
        
        sweep_results.append({
            "K": k,
            "Pair_Recall": metrics["pair_level_recall"],
            "Avg_Cands": metrics["avg_candidates_per_s1"],
            "Full_Set": metrics["full_set_recovery_rate"],
            "Total_Retrieved": int(metrics["pair_level_recall"] * sum(len(v) for v in gt_dict.values()))
        })
        
        if k == 100:
            final_cands_100 = cands
            metrics_100 = metrics
            
    print("\n--- Knee Point Analysis ---")
    print(f"{'K':<5} | {'Pair Recall':<15} | {'Avg Cands':<10} | {'Inc. Recall':<12} | {'Inc. Cands'}")
    prev_r, prev_c = sweep_results[0]["Pair_Recall"], sweep_results[0]["Avg_Cands"]
    print(f"{sweep_results[0]['K']:<5} | {prev_r:<15.4f} | {prev_c:<10.2f} | -            | -")
    for res in sweep_results[1:]:
        inc_r = res["Pair_Recall"] - prev_r
        inc_c = res["Avg_Cands"] - prev_c
        print(f"{res['K']:<5} | {res['Pair_Recall']:<15.4f} | {res['Avg_Cands']:<10.2f} | +{inc_r:<11.4f} | +{inc_c:.2f}")
        prev_r, prev_c = res["Pair_Recall"], res["Avg_Cands"]
        
    print("\n[4] Miss Analysis for K=100")
    # Identify misses at K=100
    s1_collected = s1_val_lf.collect()
    s23_collected = s23.collect()
    
    # Fast grouping
    cands_dict = {row[0]: set(row[1]) for row in final_cands_100.group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
    
    missed_pairs = []
    for s1_id, trues in gt_dict.items():
        c_set = cands_dict.get(s1_id, set())
        for t in trues:
            if t not in c_set:
                missed_pairs.append((s1_id, t))
                
    if len(missed_pairs) > 0:
        print(f"Total True Missed (K=100): {len(missed_pairs)}")
        missed_df = pl.DataFrame(missed_pairs, schema=["source1_entity_id", "matched_entity_ids"])
        mf = missed_df.join(s1_collected, left_on="source1_entity_id", right_on="entity_id", how="inner")
        mf = mf.join(s23_collected, left_on="matched_entity_ids", right_on="entity_id", how="inner", suffix="_right")
        
        miss_addr = mf.filter((pl.col("business_address") == "") | (pl.col("business_address_right") == "")).height
        short_names = mf.filter((pl.col("normalized_name").str.split(" ").list.len() <= 1) | 
                                (pl.col("normalized_name_right").str.split(" ").list.len() <= 1)).height
        
        print(f"Missing address cases: {miss_addr} ({(miss_addr/len(missed_pairs))*100:.1f}%)")
        print(f"Short/Single-token names: {short_names} ({(short_names/len(missed_pairs))*100:.1f}%)")
        print("Sample remaining misses (Name | Address):")
        for row in mf.head(5).to_dicts():
            print(f"  S1: {row['name']} | {row['business_address']}")
            print(f" S23: {row['name_right']} | {row['business_address_right']}\n")

if __name__ == "__main__":
    run_v3_rescue()
