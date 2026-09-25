import json
import polars as pl
import numpy as np

def run():
    print("Loading Task 04 splits...")
    with open("work/task04/train_val_ids.json", "r") as f:
        splits = json.load(f)
    val_s1_ids = sorted(splits["val_ids"])
    
    np.random.seed(42)
    tune_s1_ids = set(np.random.choice(val_s1_ids, size=len(val_s1_ids)//2, replace=False).tolist())
    eval_s1_ids = [x for x in val_s1_ids if x not in tune_s1_ids]
    
    print("Loading Ground Truth...")
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    gt_df = gt_df.filter(pl.col("source1_entity_id").is_in(eval_s1_ids))
    
    # Explode ground truth matched_entity_ids
    gt_df = gt_df.with_columns(
        pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id")
    ).explode("candidate_entity_id")
    gt_df = gt_df.with_columns(pl.col("candidate_entity_id").str.strip_chars())
    gt_df = gt_df.filter(pl.col("candidate_entity_id") != "")
    
    gt_pairs = set(zip(gt_df["source1_entity_id"].to_list(), gt_df["candidate_entity_id"].to_list()))
                        
    print("Loading Candidates...")
    cands_df = pl.read_parquet("work/candidates/task04_train_val_cands.parquet")
    cands_df = cands_df.filter(pl.col("source1_entity_id").is_in(eval_s1_ids))
    cands_df = cands_df.filter(pl.col("candidate_entity_id").is_not_null())
    
    cand_pairs = set(zip(cands_df["source1_entity_id"].to_list(), cands_df["candidate_entity_id"].to_list()))
            
    retrieved_gt = gt_pairs & cand_pairs
    missing_gt = gt_pairs - cand_pairs
    
    print("\n--- PHASE 1: EXACT CANDIDATE ORACLE ---")
    print(f"Eval Total GT: {len(gt_pairs)}")
    print(f"Eval Retrieved GT: {len(retrieved_gt)}")
    print(f"Candidate Recall: {len(retrieved_gt) / len(gt_pairs):.6f}" if len(gt_pairs) > 0 else "N/A")
    
    s1_cand_counts = cands_df.group_by("source1_entity_id").len()
    counts = s1_cand_counts["len"].to_list()
    print(f"Avg candidates/S1: {np.mean(counts):.2f}")
    print(f"Median candidates/S1: {np.median(counts):.2f}")
    print(f"Max candidates/S1: {np.max(counts):.2f}")
    
    print(f"\nMissing pairs: {len(missing_gt)}")
    if not missing_gt:
        return
        
    print("\n--- PHASE 2: MISSING PAIR ATTRIBUTION ---")
    missing_s1 = [x[0] for x in missing_gt]
    missing_c = [x[1] for x in missing_gt]
    missing_df = pl.DataFrame({"source1_entity_id": missing_s1, "candidate_entity_id": missing_c})
    
    from src.data.loader import load_all_data
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    
    print("Loading S23 lazy...")
    s23_lf = pl.concat([frames["train_source2"].lazy(), frames["train_source3"].lazy()], how="diagonal")
    
    print("Filtering S23 to missing targets to save memory...")
    cand_ids = missing_df.select("candidate_entity_id").unique().to_series()
    s23_filtered = s23_lf.filter(pl.col("entity_id").is_in(cand_ids)).collect()
    
    print("Joining...")
    cols = ["entity_id", "normalized_name", "normalized_address", "postal_code", "house_number", "latin_name", "latin_address"]
    df = missing_df.join(s1.select(cols), left_on="source1_entity_id", right_on="entity_id", how="left")
    df = df.join(s23_filtered.select(cols), left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_right")
    
    df = df.with_columns([
        (pl.col("normalized_name") == pl.col("normalized_name_right")).alias("exact_name"),
        (pl.col("normalized_address") == pl.col("normalized_address_right")).alias("exact_addr"),
        ((pl.col("postal_code") == pl.col("postal_code_right")) & (pl.col("house_number") == pl.col("house_number_right")) & pl.col("postal_code").is_not_null() & pl.col("house_number").is_not_null()).alias("postal_house"),
    ])
    
    c_name = df.filter(pl.col("exact_name")).height
    c_addr = df.filter(pl.col("exact_addr")).height
    c_ph = df.filter(pl.col("postal_house")).height
    
    print(f"Could be recovered by exact name (but missed!): {c_name}")
    print(f"Could be recovered by exact addr (but missed!): {c_addr}")
    print(f"Could be recovered by postal/house (but missed!): {c_ph}")

    def word_overlap(df, col1, col2):
        c1 = df[col1].to_list()
        c2 = df[col2].to_list()
        res = []
        for x, y in zip(c1, c2):
            if x and y:
                s1 = set(str(x).split())
                s2 = set(str(y).split())
                res.append(len(s1 & s2) > 0)
            else:
                res.append(False)
        return pl.Series(res, dtype=pl.Boolean)
        
    def char_overlap(df, col1, col2):
        c1 = df[col1].to_list()
        c2 = df[col2].to_list()
        res = []
        for x, y in zip(c1, c2):
            if x and y and len(str(x))>=3 and len(str(y))>=3:
                s1 = set([str(x)[i:i+3] for i in range(len(str(x))-2)])
                s2 = set([str(y)[i:i+3] for i in range(len(str(y))-2)])
                res.append(len(s1 & s2) > 0)
            else:
                res.append(False)
        return pl.Series(res, dtype=pl.Boolean)

    df = df.with_columns([
        word_overlap(df, "latin_name", "latin_name_right").alias("name_word_overlap"),
        word_overlap(df, "latin_address", "latin_address_right").alias("addr_word_overlap"),
        char_overlap(df, "latin_name", "latin_name_right").alias("name_char_overlap")
    ])
    
    c_nword = df.filter(pl.col("name_word_overlap")).height
    c_aword = df.filter(pl.col("addr_word_overlap")).height
    c_nchar = df.filter(pl.col("name_char_overlap")).height
    
    print(f"Could be recovered by Name TF-IDF (K expansion): {c_nword}")
    print(f"Could be recovered by Addr TF-IDF (K expansion): {c_aword}")
    print(f"Could be recovered by Char TF-IDF (K expansion): {c_nchar}")
    
    # Write summary for agent
    out = {
        "missing_count": len(missing_gt),
        "exact_name": c_name,
        "exact_addr": c_addr,
        "postal_house": c_ph,
        "name_word_overlap": c_nword,
        "addr_word_overlap": c_aword,
        "name_char_overlap": c_nchar
    }
    with open("work/task06/missing_pair_attribution.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    run()
