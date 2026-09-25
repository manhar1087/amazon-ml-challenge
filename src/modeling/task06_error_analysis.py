import polars as pl
import json
import os

def run_error_analysis():
    print("Loading predictions...")
    preds = pl.read_parquet("work/task04/val_predictions.parquet")
    val_s1_ids = preds["source1_entity_id"].unique().to_list()
    
    print("Loading data...")
    from src.data.loader import load_all_data
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    s23 = pl.concat([frames["train_source2"].lazy(), frames["train_source3"].lazy()], how="diagonal").collect()
    
    gt_df = frames["train_ground_truth"].collect()
    gt_dict = {}
    for row in gt_df.iter_rows(named=True):
        s_id = row["source1_entity_id"]
        if s_id in val_s1_ids:
            matches = row["matched_entity_ids"]
            if matches is None or str(matches).strip() == "":
                gt_dict[s_id] = set()
            else:
                gt_dict[s_id] = {x.strip() for x in str(matches).split(",") if x.strip()}
                
    total_true_pairs = sum(len(matches) for matches in gt_dict.values())
    
    print("Joining features for error analysis...")
    cols = ["entity_id", "country", "source", "normalized_name", "normalized_address", "postal_code", "house_number"]
    
    s1_sub = s1.select(cols)
    s23_sub = s23.select(cols)
    
    df = preds.join(s1_sub, left_on="source1_entity_id", right_on="entity_id", how="left")
    df = df.join(s23_sub, left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_right")
    
    # 1. Matcher FNs: true label = 1 but pred < 0.926
    # 2. Hard FPs: true label = 0 and pred >= 0.926
    df = df.with_columns([
        ((pl.col("label") == 1.0) & (pl.col("pred") < 0.926)).alias("is_fn_matcher"),
        ((pl.col("label") == 0.0) & (pl.col("pred") >= 0.926)).alias("is_hfp"),
        ((pl.col("label") == 1.0) & (pl.col("pred") < 0.7)).alias("is_vhfn"),
        ((pl.col("label") == 0.0) & (pl.col("pred") >= 0.95)).alias("is_vhfp")
    ])
    
    fn_matcher_df = df.filter(pl.col("is_fn_matcher"))
    hfp_df = df.filter(pl.col("is_hfp"))
    
    # Calculate Blocking FNs
    generated_pairs = {s_id: set() for s_id in val_s1_ids}
    for row in preds.iter_rows(named=True):
        c_id = row["candidate_entity_id"]
        if c_id is not None:
            generated_pairs[row["source1_entity_id"]].add(c_id)
            
    fn_blocking_count = 0
    for s_id, t_set in gt_dict.items():
        g_set = generated_pairs.get(s_id, set())
        fn_blocking_count += len(t_set - g_set)
        
    analysis = {
        "counts": {
            "total_true_pairs_in_gt": total_true_pairs,
            "fn_total": fn_matcher_df.height + fn_blocking_count,
            "fn_matcher": fn_matcher_df.height,
            "fn_blocking": fn_blocking_count,
            "hfp": hfp_df.height,
            "vhfn": df.filter(pl.col("is_vhfn")).height,
            "vhfp": df.filter(pl.col("is_vhfp")).height
        },
        "slices_matcher_fn": {}
    }
    
    def analyze_slice(col_expr):
        slice_df = df.with_columns(col_expr.alias("slice_val"))
        res = slice_df.group_by("slice_val").agg([
            pl.col("is_fn_matcher").sum().alias("fn_matcher"),
            pl.col("is_hfp").sum().alias("hfp"),
            pl.len().alias("total")
        ]).to_dicts()
        return res
        
    analysis["slices_matcher_fn"]["source23"] = analyze_slice(pl.col("source_right"))
    analysis["slices_matcher_fn"]["country"] = analyze_slice(pl.col("country"))
    analysis["slices_matcher_fn"]["exact_name"] = analyze_slice(pl.col("normalized_name") == pl.col("normalized_name_right"))
    analysis["slices_matcher_fn"]["exact_address"] = analyze_slice(pl.col("normalized_address") == pl.col("normalized_address_right"))
    
    os.makedirs("work/task06", exist_ok=True)
    with open("work/task06/error_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
        
    print("Error analysis completed and saved to work/task06/error_analysis.json")

if __name__ == '__main__':
    run_error_analysis()
