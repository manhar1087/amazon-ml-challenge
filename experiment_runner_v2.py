import time
import os
import polars as pl
import functools

print = functools.partial(print, flush=True)

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.evaluation.candidate_recall import evaluate_candidates, print_recall_report
from src.blocking.union import union_candidates
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from src.blocking.cross_script import cross_script_name_blocking
from src.blocking.sparse_retrieval import sparse_top_k_retrieval

def run_v2():
    print("Loading data for High-Recall Generation...")
    frames = load_all_data()
    
    s1 = frames["train_source1"]
    s2 = frames["train_source2"]
    s3 = frames["train_source3"]
    s23 = pl.concat([s2, s3], how="diagonal")
    
    gt = frames["train_ground_truth"].collect()
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    val_s1_set = set(val_s1_ids)
    
    s1_val_lf = s1.filter(pl.col("entity_id").is_in(list(val_s1_set)))
    
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        matches = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(matches.split(",")) if matches else set()
        
    print(f"Validation S1 entities: {len(val_s1_ids)}")
    
    blocks = []
    
    # Ex 01: Exact Name + Address + Postal/House + Cross Script
    # These are extremely fast (seconds) and highly precise.
    print("\n--- Phase 1: Exact & Structural ---")
    
    b_name = exact_name_blocking(s1_val_lf, s23)
    blocks.append(b_name)
    print_recall_report(evaluate_candidates(union_candidates(blocks), gt_dict, val_s1_ids), "1. Exact Name")
    
    b_addr = exact_address_blocking(s1_val_lf, s23)
    blocks.append(b_addr)
    print_recall_report(evaluate_candidates(union_candidates(blocks), gt_dict, val_s1_ids), "2. + Exact Address")
    
    b_ph = postal_house_blocking(s1_val_lf, s23)
    blocks.append(b_ph)
    
    b_cs = cross_script_name_blocking(s1_val_lf, s23)
    blocks.append(b_cs)
    print_recall_report(evaluate_candidates(union_candidates(blocks), gt_dict, val_s1_ids), "3. + Structural & Exact Cross-Script")
    
    # Ex 02: Token-based TF-IDF on Latin Name (Top K=20)
    print("\n--- Phase 2: Token TF-IDF (Latin Name) ---")
    t0 = time.time()
    # word-level tf-idf. max_df=0.01 drops terms appearing in >100,000 entities. 
    b_tfidf_word = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=20, channel_name="tfidf_name_word", max_df=0.05)
    print(f"Phase 2 Time: {time.time()-t0:.2f}s")
    blocks.append(b_tfidf_word)
    print_recall_report(evaluate_candidates(union_candidates(blocks), gt_dict, val_s1_ids), "4. + Token TF-IDF Name")
    
    # Ex 03: Token-based TF-IDF on Address (Top K=20)
    print("\n--- Phase 3: Token TF-IDF (Address) ---")
    t0 = time.time()
    b_tfidf_addr = sparse_top_k_retrieval(s1_val_lf, s23, col_name="normalized_address", analyzer="word", ngram_range=(1,2), k=20, channel_name="tfidf_address_word", max_df=0.1)
    print(f"Phase 3 Time: {time.time()-t0:.2f}s")
    blocks.append(b_tfidf_addr)
    print_recall_report(evaluate_candidates(union_candidates(blocks), gt_dict, val_s1_ids), "5. + Token TF-IDF Address")
    
    # Ex 04: Character N-Gram TF-IDF on Name (Top K=20)
    print("\n--- Phase 4: Character N-Gram TF-IDF (Name) ---")
    t0 = time.time()
    b_tfidf_char = sparse_top_k_retrieval(s1_val_lf, s23, col_name="latin_name", analyzer="char_wb", ngram_range=(3,4), k=20, channel_name="tfidf_name_char", max_df=0.01)
    print(f"Phase 4 Time: {time.time()-t0:.2f}s")
    blocks.append(b_tfidf_char)
    print_recall_report(evaluate_candidates(union_candidates(blocks), gt_dict, val_s1_ids), "6. + Char N-Gram TF-IDF Name")
    
    # Final union & export
    final_cands = union_candidates(blocks)
    
    os.makedirs("output", exist_ok=True)
    out_path = "output/candidate_pairs_v2.tsv"
    
    final_grouped = final_cands.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
    final_output = [{"source1_entity_id": r[0], "candidate_entity_ids": ",".join(r[1])} for r in final_grouped.iter_rows()]
    final_df = pl.DataFrame(final_output)
    final_df.write_csv(out_path, separator="\t")
    print(f"\nFinal candidate set saved to {out_path} with {final_df.height} rows.")

if __name__ == "__main__":
    run_v2()
