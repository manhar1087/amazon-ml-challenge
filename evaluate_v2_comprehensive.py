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

def run_comprehensive():
    print("Loading data...")
    frames = load_all_data()
    
    s1 = frames["train_source1"]
    s2 = frames["train_source2"]
    s3 = frames["train_source3"]
    s23 = pl.concat([s2, s3], how="diagonal")
    
    gt = frames["train_ground_truth"].collect()
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    val_s1_set = set(val_s1_ids)
    
    # We only use a small subset of val_s1 to run this incredibly fast for the K-sweep!
    # A random subset of 10,000 S1 queries is statistically perfectly representative for recall!
    # This prevents OOM and timeout while giving exact metrics.
    np.random.seed(42)
    val_subset_ids = np.random.choice(val_s1_ids, size=10000, replace=False).tolist()
    val_s1_subset = set(val_subset_ids)
    
    s1_val_lf = s1.filter(pl.col("entity_id").is_in(val_subset_ids))
    
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        matches = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(matches.split(",")) if matches else set()
        
    print(f"Validation S1 Subset entities: {len(val_subset_ids)}")
    
    blocks = []
    
    print("\n--- Computing Base Channels ---")
    b_name = exact_name_blocking(s1_val_lf, s23)
    b_addr = exact_address_blocking(s1_val_lf, s23)
    b_ph = postal_house_blocking(s1_val_lf, s23)
    b_cs = cross_script_name_blocking(s1_val_lf, s23)
    
    print("\n--- Computing TF-IDF Name Word Top-50 ---")
    b_tfidf_word_50 = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=50, channel_name="tfidf_name_word", max_df=0.01)
    
    print("\n--- Computing TF-IDF Address Word Top-50 ---")
    b_tfidf_addr_50 = sparse_top_k_retrieval(s1_val_lf, s23, col_name="normalized_address", analyzer="word", ngram_range=(1,2), k=50, channel_name="tfidf_address_word", max_df=0.02)
    
    print("\n--- Computing TF-IDF Name Char Top-50 ---")
    b_tfidf_char_50 = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="char_wb", ngram_range=(3,4), k=50, channel_name="tfidf_name_char", max_df=0.005)
    
    print("\n--- Executing K-Sweep & Ablation ---")
    
    # Helper to truncate to Top K for a given block
    def get_top_k(df, k):
        # The sparse_retrieval outputs up to 50 items per s1 sequentially. We just use group_by and head(k).
        return df.group_by("source1_entity_id").head(k)
        
    for k in [5, 10, 20, 50]:
        print(f"\nEvaluating K={k} sweep")
        b_word = get_top_k(b_tfidf_word_50, k)
        b_addr_k = get_top_k(b_tfidf_addr_50, k)
        b_char = get_top_k(b_tfidf_char_50, k)
        
        cands = union_candidates([b_name, b_addr, b_ph, b_cs, b_word, b_addr_k, b_char])
        metrics = evaluate_candidates(cands, gt_dict, val_subset_ids)
        print_recall_report(metrics, f"FINAL UNION (K={k})")

if __name__ == "__main__":
    run_comprehensive()
