import time
import os
import polars as pl
import json
import functools

print = functools.partial(print, flush=True)

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.evaluation.candidate_recall import evaluate_candidates, print_recall_report

from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from src.blocking.cross_script import cross_script_name_blocking
from src.blocking.token import token_blocking
from src.blocking.union import union_candidates

def run_experiments():
    print("Loading data...")
    frames = load_all_data()
    
    s1 = frames["train_source1"]
    s2 = frames["train_source2"]
    s3 = frames["train_source3"]
    
    # Combine S2/S3
    s23 = pl.concat([s2, s3], how="diagonal")
    
    # Load ground truth
    gt = frames["train_ground_truth"].collect()
    
    print("Preparing validation split...")
    s1_ids_all = gt["source1_entity_id"].to_list()
    # Use split function to ensure determinism
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    print(f"Validation S1 entities: {len(val_s1_ids)}")
    
    # Only block using the validation set to save time and memory for experiments
    # Wait, token IDF should ideally be on full S2/S3, but blocking is only requested for val_s1.
    val_s1_set = set(val_s1_ids)
    s1_val_lf = s1.filter(pl.col("entity_id").is_in(list(val_s1_set)))
    
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        matches = row["matched_entity_ids"]
        if matches:
            gt_dict[row["source1_entity_id"]] = set(matches.split(","))
        else:
            gt_dict[row["source1_entity_id"]] = set()
            
    print("Starting experiments...")
    results = []
    
    blocks = []
    
    # Ex 01: Exact Name
    print("\n--- Experiment 01: Exact Name ---")
    t0 = time.time()
    b1 = exact_name_blocking(s1_val_lf, s23)
    t_ex1 = time.time() - t0
    blocks.append(b1)
    cands_ex1 = union_candidates(blocks)
    metrics_ex1 = evaluate_candidates(cands_ex1, gt_dict, val_s1_ids)
    print_recall_report(metrics_ex1, "Ex 01: Exact Name")
    print(f"Runtime: {t_ex1:.2f}s")
    
    # Ex 02: + Exact Address
    print("\n--- Experiment 02: + Exact Address ---")
    t0 = time.time()
    b2 = exact_address_blocking(s1_val_lf, s23)
    t_ex2 = time.time() - t0
    blocks.append(b2)
    cands_ex2 = union_candidates(blocks)
    metrics_ex2 = evaluate_candidates(cands_ex2, gt_dict, val_s1_ids)
    print_recall_report(metrics_ex2, "Ex 02: + Exact Address")
    print(f"Runtime: {t_ex2:.2f}s")
    
    # Ex 03: + Structural (Postal + House)
    print("\n--- Experiment 03: + Structural Blocks ---")
    t0 = time.time()
    b3 = postal_house_blocking(s1_val_lf, s23)
    t_ex3 = time.time() - t0
    blocks.append(b3)
    cands_ex3 = union_candidates(blocks)
    metrics_ex3 = evaluate_candidates(cands_ex3, gt_dict, val_s1_ids)
    print_recall_report(metrics_ex3, "Ex 03: + Structural (Postal+House)")
    print(f"Runtime: {t_ex3:.2f}s")
    
    # Ex 04: + Cross-script (Latin Name)
    print("\n--- Experiment 04: + Cross-script (Latin Name) ---")
    t0 = time.time()
    b4 = cross_script_name_blocking(s1_val_lf, s23)
    t_ex4 = time.time() - t0
    blocks.append(b4)
    cands_ex4 = union_candidates(blocks)
    metrics_ex4 = evaluate_candidates(cands_ex4, gt_dict, val_s1_ids)
    print_recall_report(metrics_ex4, "Ex 04: + Cross-script")
    print(f"Runtime: {t_ex4:.2f}s")
    
    # Ex 05: + Token Retrieval (Shared >= 2 tokens)
    print("\n--- Experiment 05: + Token Retrieval (Shared >= 2) ---")
    t0 = time.time()
    b5 = token_blocking(s1_val_lf, s23, min_shared_tokens=2)
    t_ex5 = time.time() - t0
    blocks.append(b5)
    cands_ex5 = union_candidates(blocks)
    metrics_ex5 = evaluate_candidates(cands_ex5, gt_dict, val_s1_ids)
    print_recall_report(metrics_ex5, "Ex 05: + Token Retrieval")
    print(f"Runtime: {t_ex5:.2f}s")
    
    # Ex 06: + N-Gram Signature (Consonants)
    print("\n--- Experiment 06: + N-Gram/Typo Signature ---")
    t0 = time.time()
    from src.blocking.ngram import ngram_signature_blocking
    b6 = ngram_signature_blocking(s1_val_lf, s23)
    t_ex6 = time.time() - t0
    blocks.append(b6)
    cands_ex6 = union_candidates(blocks)
    metrics_ex6 = evaluate_candidates(cands_ex6, gt_dict, val_s1_ids)
    print_recall_report(metrics_ex6, "Ex 06: + N-Gram Signature")
    print(f"Runtime: {t_ex6:.2f}s")
    
    # Final Output
    os.makedirs("output", exist_ok=True)
    out_path = "output/candidate_pairs.tsv"
    
    # Convert to expected submission format
    final_grouped = cands_ex6.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
    
    final_output = []
    for row in final_grouped.iter_rows():
        s1_id = row[0]
        cands = list(row[1])
        final_output.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands)
        })
        
    final_df = pl.DataFrame(final_output)
    final_df.write_csv(out_path, separator="\t")
    print(f"\nFinal candidate set saved to {out_path} with {final_df.height} rows.")

if __name__ == "__main__":
    run_experiments()
