import json
import polars as pl
import os

def slice_v3_errors():
    print("Loading V3 raw predictions...")
    preds = pl.read_parquet("work/task06/v3_raw_eval_predictions.parquet")
    val_ids = preds["source1_entity_id"].unique().to_list()
    
    print(f"Validation S1s in predictions: {len(val_ids)}")
    
    # We will simulate the Oracle on this exact subset
    from src.data.loader import load_all_data
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    s23 = pl.concat([frames["train_source2"].lazy(), frames["train_source3"].lazy()], how="diagonal").collect()
    
    gt_df = frames["train_ground_truth"].collect()
    gt_dict = {}
    for row in gt_df.iter_rows(named=True):
        s_id = row["source1_entity_id"]
        if s_id in val_ids:
            matches = row["matched_entity_ids"]
            if matches is not None and str(matches).strip() != "":
                gt_dict[s_id] = {x.strip() for x in str(matches).split(",") if x.strip()}
            else:
                gt_dict[s_id] = set()
                
    total_gt = sum(len(matches) for matches in gt_dict.values())
    
    cands_df = pl.read_parquet("work/candidates/task04_train_val_cands.parquet")
    cand_pairs = set()
    for row in cands_df.select(["source1_entity_id", "candidate_entity_id"]).iter_rows():
        s_id, c_id = row
        if s_id in val_ids and c_id is not None:
            cand_pairs.add((s_id, c_id))
            
    gt_pairs = set()
    for s_id, matches in gt_dict.items():
        for c_id in matches:
            gt_pairs.add((s_id, c_id))
            
    retrieved_gt = len(gt_pairs & cand_pairs)
    print(f"\n--- Candidate-Limited Oracle ---")
    print(f"Eval Total GT: {total_gt}")
    print(f"Eval Retrieved GT: {retrieved_gt}")
    print(f"Candidate Recall (Oracle): {retrieved_gt / total_gt:.6f}" if total_gt > 0 else "N/A")
    
    print("\nJoining features...")
    cols = ["entity_id", "country", "source", "normalized_name", "normalized_address", "postal_code", "house_number"]
    
    s1_sub = s1.select(cols)
    s23_sub = s23.select(cols)
    
    df = preds.join(s1_sub, left_on="source1_entity_id", right_on="entity_id", how="left")
    df = df.join(s23_sub, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_right")
    
    df = df.with_columns([
        pl.struct(["source1_entity_id", "candidate_entity_id"]).map_batches(
            lambda s: pl.Series([1.0 if r["candidate_entity_id"] in gt_dict.get(r["source1_entity_id"], set()) else 0.0 for r in s.to_list()], dtype=pl.Float32),
            return_dtype=pl.Float32
        ).alias("label")
    ])
    
    # Slicing
    with open("work/task06/model_comparison_v3.json", "r") as f:
        comp = json.load(f)
    best_th = comp["Binary_V3"]["best_tuning_policy"]["policy"].get("threshold", 0.5)
    
    df = df.with_columns([
        ((pl.col("label") == 1.0) & (pl.col("pred") < best_th)).alias("is_fn"),
        ((pl.col("label") == 0.0) & (pl.col("pred") >= best_th)).alias("is_hfp"),
    ])
    
    analysis = {
        "oracle": {
            "total_gt": total_gt,
            "retrieved_gt": retrieved_gt,
            "recall": retrieved_gt / total_gt if total_gt > 0 else 0
        },
        "slices": {}
    }
    
    def analyze_slice(col_expr):
        slice_df = df.with_columns(col_expr.alias("slice_val"))
        res = slice_df.group_by("slice_val").agg([
            pl.col("is_fn").sum().alias("fn"),
            pl.col("is_hfp").sum().alias("hfp"),
            pl.len().alias("total")
        ]).to_dicts()
        return res
        
    analysis["slices"]["source23"] = analyze_slice(pl.col("source_right"))
    analysis["slices"]["country"] = analyze_slice(pl.col("country"))
    analysis["slices"]["exact_name"] = analyze_slice(pl.col("normalized_name") == pl.col("normalized_name_right"))
    analysis["slices"]["exact_address"] = analyze_slice(pl.col("normalized_address") == pl.col("normalized_address_right"))
    analysis["slices"]["postal_match"] = analyze_slice(pl.col("postal_code") == pl.col("postal_code_right"))
    analysis["slices"]["house_match"] = analyze_slice(pl.col("house_number") == pl.col("house_number_right"))
    
    os.makedirs("work/task06", exist_ok=True)
    with open("work/task06/v3_error_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
        
    print("V3 Error analysis completed and saved to work/task06/v3_error_analysis.json")

if __name__ == "__main__":
    slice_v3_errors()
