import polars as pl

def eval_channels():
    cands = pl.read_parquet("work/candidates/task03_frozen_holdout_20k.parquet").drop_nulls()
    from src.data.loader import load_all_data
    gt = load_all_data()["train_ground_truth"].collect()
    gt_pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "")
    
    # We only care about 20k holdout
    s1_ids = cands["source1_entity_id"].unique()
    gt_pairs = gt_pairs.filter(pl.col("source1_entity_id").is_in(s1_ids))
    gt_set = set(zip(gt_pairs["source1_entity_id"], gt_pairs["matched_entity_ids"]))
    
    channels = ["exact_name", "exact_addr", "postal_house", "cross_script", "abbr", "tfidf_name_k50", "tfidf_addr_k20", "tfidf_char_k20"]
    
    cumulative_cands = set()
    cumulative_matches = set()
    
    print("\n--- Channel Incremental Report ---")
    for ch in channels:
        ch_cands = cands.filter(pl.col("retrieval_channels").str.contains(ch))
        ch_pairs = set(zip(ch_cands["source1_entity_id"], ch_cands["candidate_entity_id"]))
        
        ch_matches = ch_pairs & gt_set
        
        added_cands = ch_pairs - cumulative_cands
        added_matches = ch_matches - cumulative_matches
        
        cumulative_cands.update(ch_pairs)
        cumulative_matches.update(ch_matches)
        
        print(f"Channel: {ch}")
        print(f"  Candidates generated: {len(ch_pairs)}")
        print(f"  Unique true matches recovered: {len(ch_matches)}")
        print(f"  Incremental volume added: {len(added_cands)}")
        print(f"  Incremental true matches: {len(added_matches)}\n")
        
if __name__ == "__main__":
    eval_channels()
