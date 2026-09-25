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
from src.blocking.abbreviation import abbreviation_blocking
from sklearn.feature_extraction.text import TfidfVectorizer

def fast_sparse_top_k(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame, 
                           col_name="latin_name", 
                           analyzer="word", 
                           ngram_range=(1,1),
                           k=20,
                           channel_name="tfidf",
                           chunk_size=10000,
                           min_df=2,
                           max_df=0.01):
    s1 = s1_lf.select(["entity_id", col_name]).collect().filter(pl.col(col_name) != "")
    s23 = s2_s3_lf.select(["entity_id", col_name]).collect().filter(pl.col(col_name) != "")
    
    print(f"[{channel_name}] Vectorizer...")
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_df=max_df, dtype=np.float32)
    
    print(f"[{channel_name}] Fit Transform S23...")
    X_s23 = vectorizer.fit_transform(s23[col_name].to_list())
    
    print(f"[{channel_name}] Transform S1...")
    X_s1 = vectorizer.transform(s1[col_name].to_list())
    
    num_queries = X_s1.shape[0]
    num_chunks = int(np.ceil(num_queries / chunk_size))
    
    s1_ids = s1["entity_id"].to_numpy()
    s23_ids = s23["entity_id"].to_numpy()
    
    out_s1 = []
    out_s23 = []
    
    print(f"[{channel_name}] Computing dot products for {num_chunks} chunks...")
    t0 = time.time()
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, num_queries)
        
        sim = X_s1[start:end].dot(X_s23.T)
        
        for row_idx in range(sim.shape[0]):
            row_data = sim.data[sim.indptr[row_idx]:sim.indptr[row_idx+1]]
            row_indices = sim.indices[sim.indptr[row_idx]:sim.indptr[row_idx+1]]
            if len(row_data) > 0:
                if len(row_data) > k:
                    top_k_idx = np.argpartition(row_data, -k)[-k:]
                    top_k_indices = row_indices[top_k_idx]
                else:
                    top_k_indices = row_indices
                
                out_s1.extend([s1_ids[start + row_idx]] * len(top_k_indices))
                out_s23.extend(s23_ids[top_k_indices])
                
        if (i+1) % 5 == 0:
            print(f"[{channel_name}] Chunk {i+1}/{num_chunks}")
            
    df = pl.DataFrame({
        "source1_entity_id": out_s1,
        "candidate_entity_id": out_s23,
        "retrieval_channels": [channel_name] * len(out_s1)
    })
    return df

def run_v4_full():
    print("Loading data for Full Validation...")
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    _, val_s1_ids = get_grouped_split(s1_ids_all)
    val_s1_set = set(val_s1_ids)
    
    s1_val_lf = s1.filter(pl.col("entity_id").is_in(val_s1_ids))
    
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        if row["source1_entity_id"] in val_s1_set:
            matches = row["matched_entity_ids"]
            gt_dict[row["source1_entity_id"]] = set(matches.split(",")) if matches else set()
            
    print("\n[1] Computing Base Channels...")
    b_name = exact_name_blocking(s1_val_lf, s23)
    b_addr = exact_address_blocking(s1_val_lf, s23)
    b_ph = postal_house_blocking(s1_val_lf, s23)
    b_cs = cross_script_name_blocking(s1_val_lf, s23)
    b_abbr = abbreviation_blocking(s1_val_lf, s23)
    
    print("\n[2] Computing Top-K TF-IDF Channels (Full 441k)...")
    # Tighter parameters to finish fast
    # Name K=50, Address K=20, Char K=20
    b_word = fast_sparse_top_k(s1_val_lf, s23, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=50, channel_name="tfidf_name", chunk_size=5000, min_df=5, max_df=0.01)
    b_addr_tf = fast_sparse_top_k(s1_val_lf, s23, col_name="normalized_address", analyzer="word", ngram_range=(1,1), k=20, channel_name="tfidf_addr", chunk_size=10000, min_df=10, max_df=0.01)
    b_char = fast_sparse_top_k(s1_val_lf, s23, col_name="latin_name", analyzer="char_wb", ngram_range=(3,4), k=20, channel_name="tfidf_char", chunk_size=5000, min_df=10, max_df=0.005)
    
    cands = union_candidates([b_name, b_addr, b_ph, b_cs, b_abbr, b_word, b_addr_tf, b_char])
    
    print("\n[3] Evaluating Final Union...")
    metrics = evaluate_candidates(cands, gt_dict, val_s1_ids)
    print_recall_report(metrics, "FULL VALIDATION (Name=50, Addr=20, Char=20)")
    
    # Save candidate pairs
    out_path = "output/candidate_pairs.tsv"
    final_grouped = cands.group_by("source1_entity_id").agg(pl.col("candidate_entity_id"))
    final_output = [{"source1_entity_id": r[0], "candidate_entity_ids": ",".join(r[1])} for r in final_grouped.iter_rows()]
    pl.DataFrame(final_output).write_csv(out_path, separator="\t")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_v4_full()
