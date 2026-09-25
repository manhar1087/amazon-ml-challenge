import json
import polars as pl

def audit():
    with open("work/task04/train_val_ids.json", "r") as f:
        val_ids = set(json.load(f)["val_ids"])
        
    print(f"Validation S1s: {len(val_ids)}")
    
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    gt_pairs = set()
    for row in gt_df.iter_rows(named=True):
        s1 = row["source1_entity_id"]
        if s1 in val_ids:
            matches = row["matched_entity_ids"]
            if matches is not None and str(matches).strip() != "":
                for c23 in str(matches).split(","):
                    gt_pairs.add((s1, c23.strip()))
                    
    total_gt = len(gt_pairs)
    print(f"1. Total ground-truth positive pairs: {total_gt}")
    
    cands_df = pl.read_parquet("work/candidates/task04_train_val_cands.parquet")
    cand_pairs = set()
    for row in cands_df.select(["source1_entity_id", "candidate_entity_id"]).iter_rows():
        s1, c23 = row
        if s1 in val_ids and c23 is not None:
            cand_pairs.add((s1, c23))
            
    retrieved_gt = len(gt_pairs & cand_pairs)
    blocking_fn = total_gt - retrieved_gt
    
    print(f"2. Positive pairs present in candidate artifact: {retrieved_gt}")
    print(f"3. Positive pairs absent from candidate artifact (Blocking FNs): {blocking_fn}")
    
    preds_df = pl.read_parquet("work/task04/val_predictions.parquet")
    tp_pairs = set()
    fn_pairs = set()
    for row in preds_df.select(["source1_entity_id", "candidate_entity_id", "pred"]).iter_rows():
        s1, c23, pred = row
        if s1 in val_ids and c23 is not None and (s1, c23) in gt_pairs:
            if pred >= 0.926:
                tp_pairs.add((s1, c23))
            else:
                fn_pairs.add((s1, c23))
                
    matcher_tp = len(tp_pairs)
    matcher_fn = len(fn_pairs)
    
    print(f"4. Positive pairs present AND pred >= 0.926: {matcher_tp}")
    print(f"5. Positive pairs present AND pred < 0.926: {matcher_fn}")
    
    print("\n--- Verification ---")
    print(f"total_gt ({total_gt}) == retrieved_gt ({retrieved_gt}) + blocking_fn ({blocking_fn}) -> {total_gt == retrieved_gt + blocking_fn}")
    print(f"retrieved_gt ({retrieved_gt}) == matcher_tp ({matcher_tp}) + matcher_fn ({matcher_fn}) -> {retrieved_gt == matcher_tp + matcher_fn}")
    
    print("\n--- Recalls ---")
    print(f"Candidate pair recall: {retrieved_gt / total_gt:.6f}")
    print(f"Matcher recall conditional on retrieval: {matcher_tp / retrieved_gt:.6f}")
    print(f"Overall pair recall at threshold 0.926: {matcher_tp / total_gt:.6f}")
    
if __name__ == "__main__":
    audit()
