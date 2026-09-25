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

def run_mixed_k():
    print("Loading data for Mixed-K Analysis...")
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    
    np.random.seed(102)
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
    
    print("\n[2] Computing Top-50 TF-IDF Channels...")
    b_word = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=50, channel_name="tfidf_name_word", max_df=0.01)
    b_addr_tf = sparse_top_k_retrieval(s1_val_lf, s23, col_name="normalized_address", analyzer="word", ngram_range=(1,2), k=50, channel_name="tfidf_address_word", max_df=0.02)
    b_char = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="char_wb", ngram_range=(3,4), k=50, channel_name="tfidf_name_char", max_df=0.005)
    
    def slice_k(df, k):
        return df.group_by("source1_entity_id").head(k)
        
    configs = [
        {"name": 50, "addr": 20, "char": 20},
        {"name": 50, "addr": 20, "char": 50},
        {"name": 50, "addr": 50, "char": 20},
        {"name": 50, "addr": 50, "char": 50},
    ]
    
    for c in configs:
        print(f"\n--- Config: Name={c['name']}, Addr={c['addr']}, Char={c['char']} ---")
        w_k = slice_k(b_word, c["name"])
        a_k = slice_k(b_addr_tf, c["addr"])
        c_k = slice_k(b_char, c["char"])
        
        cands = union_candidates([b_name, b_addr, b_ph, b_cs, b_abbr, w_k, a_k, c_k])
        metrics = evaluate_candidates(cands, gt_dict, val_subset_ids)
        print_recall_report(metrics, f"Mixed K ({c['name']}/{c['addr']}/{c['char']})")

if __name__ == "__main__":
    run_mixed_k()
