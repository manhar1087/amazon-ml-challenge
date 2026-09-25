import polars as pl
from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
import numpy as np

def run_diagnosis():
    print("Loading data for diagnosis...")
    frames = load_all_data()
    
    s1 = frames["train_source1"].collect()
    s23 = pl.concat([frames["train_source2"].collect(), frames["train_source3"].collect()], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    val_s1_set = set(val_s1_ids)
    
    # Load candidates
    try:
        cands_df = pl.read_csv("output/candidate_pairs.tsv", separator="\t")
    except Exception as e:
        print(f"Could not load candidates: {e}")
        return
        
    cands_dict = {}
    for row in cands_df.iter_rows(named=True):
        cands = row["candidate_entity_ids"]
        cands_dict[row["source1_entity_id"]] = set(cands.split(",")) if cands else set()
        
    # Get True Matches
    gt_exploded = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .filter(pl.col("source1_entity_id").is_in(list(val_s1_set)))
    )
    
    # Find Missed Matches
    gt_pairs = gt_exploded.select(["source1_entity_id", "matched_entity_ids"]).to_dicts()
    
    missed_pairs = []
    found_pairs = []
    
    for row in gt_pairs:
        s1_id = row["source1_entity_id"]
        s23_id = row["matched_entity_ids"]
        
        if s23_id in cands_dict.get(s1_id, set()):
            found_pairs.append((s1_id, s23_id))
        else:
            missed_pairs.append((s1_id, s23_id))
            
    print(f"Found {len(found_pairs)} true matches.")
    print(f"Missed {len(missed_pairs)} true matches.")
    
    if len(missed_pairs) == 0:
        return
        
    missed_df = pl.DataFrame(missed_pairs, schema=["source1_entity_id", "matched_entity_ids"])
    
    # Join features
    missed_features = missed_df.join(s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
    missed_features = missed_features.join(s23, left_on="matched_entity_ids", right_on="entity_id", how="inner", suffix="_right")
    
    # Categorize misses
    # 1. Missing address
    missed_addr = missed_features.filter((pl.col("business_address") == "") | (pl.col("business_address_right") == "")).height
    
    # 2. Structural mismatch
    # Have addresses, but postal/house differ
    has_addr = missed_features.filter((pl.col("business_address") != "") & (pl.col("business_address_right") != ""))
    diff_postal = has_addr.filter(pl.col("postal_code") != pl.col("postal_code_right")).height
    diff_house = has_addr.filter(pl.col("house_number") != pl.col("house_number_right")).height
    
    # 3. Transliteration / Cross-script (different scripts, or large unidecode difference)
    # 4. Short single token names
    short_names = missed_features.filter((pl.col("normalized_name").str.split(" ").list.len() <= 1) | 
                                         (pl.col("normalized_name_right").str.split(" ").list.len() <= 1)).height
                                         
    # 5. Shared tokens = 0 or 1
    def count_shared_tokens(col1, col2):
        # A simple approximation
        pass
        
    total_missed = len(missed_pairs)
    print(f"\nBreakdown of {total_missed} missed matches:")
    print(f"Missing address in either S1 or S2/S3: {missed_addr} ({(missed_addr/total_missed)*100:.1f}%)")
    print(f"Short/Single token name in either: {short_names} ({(short_names/total_missed)*100:.1f}%)")
    print(f"Different Postal Code (when both have address): {diff_postal} ({(diff_postal/total_missed)*100:.1f}%)")
    print(f"Different House Number (when both have address): {diff_house} ({(diff_house/total_missed)*100:.1f}%)")
    
    # By source
    s2_misses = missed_features.filter(pl.col("matched_entity_ids").str.starts_with("S2")).height
    s3_misses = missed_features.filter(pl.col("matched_entity_ids").str.starts_with("S3")).height
    print(f"\nS1 -> S2 Misses: {s2_misses}")
    print(f"S1 -> S3 Misses: {s3_misses}")

if __name__ == "__main__":
    run_diagnosis()
