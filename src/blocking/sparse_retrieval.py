import polars as pl
import numpy as np
import warnings
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os

def get_deterministic_top_k(scores, global_indices, k):
    if len(scores) <= k:
        order = np.lexsort((global_indices, -scores))
        return scores[order], global_indices[order]
    kth_idx = np.argpartition(-scores, k - 1)[k - 1]
    boundary_score = scores[kth_idx]
    candidates = np.where(scores >= boundary_score)[0]
    order = np.lexsort((global_indices[candidates], -scores[candidates]))
    best_idx = candidates[order[:k]]
    return scores[best_idx], global_indices[best_idx]

def sparse_top_k_retrieval(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame, 
                           col_name="latin_name", 
                           analyzer="word", 
                           ngram_range=(1,1),
                           k=20,
                           channel_name="tfidf_word",
                           chunk_size=500,
                           max_df=0.1) -> pl.DataFrame:
    """
    DEPRECATED COMPATIBILITY PATH
    
    This function was the legacy monolithic TF-IDF candidate generator. It is
    unsafe for large corpora (e.g. 10M rows) as it builds the entire vocabulary 
    and document frequency matrix in memory and performs unrestricted dense 
    similarity extraction.
    
    The sp_matmul_topn dependency has been completely removed because it randomly
    truncates tied candidates. This compatibility path now uses an exact, 
    unrestricted dot product (X @ C.T) which guarantees correct tie-handling 
    but will OOM on large datasets.
    
    MIGRATE TO: generate_task03_holdout.py -> run_tfidf_chunked()
    """
    warnings.warn(
        "sparse_top_k_retrieval is deprecated and unsafe for production scale. "
        "It uses an unrestricted similarity matrix. Please migrate to run_tfidf_chunked.",
        DeprecationWarning, stacklevel=2
    )
    
    # Collect necessary columns
    s1 = s1_lf.select(["entity_id", col_name]).collect()
    s23 = s2_s3_lf.select(["entity_id", col_name]).collect()
    
    # Filter empty
    s1 = s1.filter(pl.col(col_name) != "")
    s23 = s23.filter(pl.col(col_name) != "")
    
    if s23.height > 1_000_000:
        warnings.warn(f"Running deprecated sparse_top_k_retrieval on {s23.height} rows. OOM risk is very high.")
    
    print(f"[{channel_name}] Building TF-IDF Vectorizer (analyzer={analyzer}, ngram={ngram_range})...")
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2, max_df=max_df, dtype=np.float32)
    
    print(f"[{channel_name}] Fitting S2/S3 Corpus...")
    X_s23 = vectorizer.fit_transform(s23[col_name].to_list())
    
    print(f"[{channel_name}] Transforming S1 Queries...")
    X_s1 = vectorizer.transform(s1[col_name].to_list())
    
    s1_ids = s1["entity_id"].to_numpy()
    s23_ids = s23["entity_id"].to_numpy()
    
    num_queries = X_s1.shape[0]
    num_chunks = int(np.ceil(num_queries / chunk_size))
    
    out_s1 = []
    out_s23 = []
    
    print(f"[{channel_name}] Processing {num_chunks} chunks for Top-{k} retrieval...")
    t0 = time.time()
    
    C_T = X_s23.T.tocsr()
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i+1) * chunk_size, num_queries)
        
        # Exact unrestricted sparse dot product. 
        # Safely extracts all ties without sp_matmul_topn truncation risk.
        sim = (X_s1[start:end] @ C_T)
        
        for row_idx in range(sim.shape[0]):
            ptr_start = sim.indptr[row_idx]
            ptr_end = sim.indptr[row_idx+1]
            row_data = sim.data[ptr_start:ptr_end]
            row_indices = sim.indices[ptr_start:ptr_end]
            
            if len(row_data) > 0:
                bs, bi = get_deterministic_top_k(row_data, row_indices, k)
                s1_id = s1_ids[start + row_idx]
                out_s1.extend([s1_id] * len(bi))
                out_s23.extend(s23_ids[bi])
                
        if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[{channel_name}] Chunk {i+1}/{num_chunks} done. Rate: {rate:.2f} chunks/sec")
            
    df = pl.DataFrame({
        "source1_entity_id": out_s1,
        "candidate_entity_id": out_s23,
        "retrieval_channels": [channel_name] * len(out_s1)
    })
    
    return df

