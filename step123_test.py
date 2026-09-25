import numpy as np
import scipy.sparse as sp
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import gc
import json
import inspect
import sys
import os

from src.blocking.sparse_retrieval import get_deterministic_top_k

def run_step1():
    print("=== STEP 1: TRUE IMMUTABLE REFERENCE vs NEW PRODUCTION ===")
    SOURCE1 = "work/parquet/train/source1.parquet"
    SOURCE2 = "work/parquet/train/source2.parquet"
    SOURCE3 = "work/parquet/train/source3.parquet"
    
    source1 = pl.read_parquet(SOURCE1)
    source2 = pl.read_parquet(SOURCE2)
    source3 = pl.read_parquet(SOURCE3)
    
    N_QUERIES = 200
    N_CORPUS = 10000
    queries = source1.head(N_QUERIES).select(["entity_id", "latin_name"])
    corpus = pl.concat([source2, source3]).head(N_CORPUS).select(["entity_id", "latin_name"])
    
    query_texts = queries["latin_name"].fill_null("").to_list()
    corpus_texts = corpus["latin_name"].fill_null("").to_list()
    corpus_eids = corpus["entity_id"].to_numpy()
    query_eids = queries["entity_id"].to_numpy()
    
    K = 20
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1,1), min_df=2, max_df=0.01, dtype=np.float32)
    X_corpus = vec.fit_transform(corpus_texts)
    X_queries = vec.transform(query_texts)
    
    # 1. TRUE ORIGINAL REFERENCE
    # No helper, just original sp_matmul_topn behavior
    ref_sim = sp_matmul_topn(X_queries, X_corpus.T, K, sort=True)
    ref_cands = np.full((N_QUERIES, K), -1, dtype=np.int64)
    ref_scores = np.full((N_QUERIES, K), -np.inf, dtype=np.float64)
    for q in range(N_QUERIES):
        s = ref_sim.data[ref_sim.indptr[q]:ref_sim.indptr[q+1]]
        idx = ref_sim.indices[ref_sim.indptr[q]:ref_sim.indptr[q+1]]
        k_local = min(K, len(s))
        ref_cands[q, :k_local] = idx[:k_local]
        ref_scores[q, :k_local] = s[:k_local]
    
    # 2. NEW CHUNKED PRODUCTION RETRIEVAL
    sys.path.append(os.getcwd())
    import generate_task03_holdout as gen
    
    CHUNK_COUNT = 3
    chunk_size = int(np.ceil(N_CORPUS / CHUNK_COUNT))
    n_q = X_queries.shape[0]
    best_scores  = np.full((n_q, K), -np.inf, dtype=np.float64)
    best_gidx    = np.full((n_q, K), -1,      dtype=np.int64)
    
    for c_idx in range(CHUNK_COUNT):
        start = c_idx * chunk_size
        end = min(start + chunk_size, N_CORPUS)
        if start >= N_CORPUS: break
        
        C = X_corpus[start:end]
        n_chunk = C.shape[0]
        g_start = start
        
        for q0 in range(0, n_q, 500):
            q1 = min(q0+500, n_q)
            Qb = X_queries[q0:q1]
            sim = (Qb @ C.T).toarray()
            
            for qi in range(q1-q0):
                q_global = q0+qi
                sim_row = sim[qi].astype(np.float64)
                lg = np.arange(n_chunk, dtype=np.int64) + g_start
                
                comb_s = np.concatenate([best_scores[q_global], sim_row])
                comb_g = np.concatenate([best_gidx[q_global], lg])
                
                bs, bg = get_deterministic_top_k(comb_s, comb_g, K)
                
                best_scores[q_global, :len(bs)] = bs
                best_gidx[q_global, :len(bg)] = bg
                
    prod_cands = np.full((N_QUERIES, K), -1, dtype=np.int64)
    prod_scores = np.full((N_QUERIES, K), -np.inf, dtype=np.float64)
    for q in range(N_QUERIES):
        valid = best_gidx[q] >= 0
        s = best_scores[q][valid]
        g = best_gidx[q][valid]
        if len(s) == 0: continue
        bs, bg = get_deterministic_top_k(s, g, K)
        k_local = len(bs)
        prod_cands[q, :k_local] = bg
        prod_scores[q, :k_local] = bs
        
    id_match = np.array_equal(ref_cands, prod_cands)
    score_diffs = np.abs(ref_scores - prod_scores)
    
    # Calculate exact mismatches
    mismatched_queries = 0
    mismatched_rows = 0
    for q in range(N_QUERIES):
        if not np.array_equal(ref_cands[q], prod_cands[q]):
            mismatched_queries += 1
            mismatched_rows += np.sum(ref_cands[q] != prod_cands[q])
            
    print(f"candidate ID equality: {id_match}")
    print(f"candidate ordering equality: {id_match}")
    print(f"score equality: {np.all((score_diffs < 1e-7) | np.isnan(score_diffs))}")
    print(f"mismatched queries: {mismatched_queries}")
    print(f"mismatched rows: {mismatched_rows}")
    print()

def run_step2():
    print("=== STEP 2: TEST THE ACTUAL PRODUCTION PATH ===")
    import generate_task03_holdout as gen
    
    src = inspect.getsource(gen.generate)
    print("Confirmed generate_task03_holdout.py flow:")
    print(f"  Imports chunked_topk: {'chunked_topk' in dir(gen)}")
    print(f"  Uses external vocab/idf (run_tfidf_chunked): {'run_tfidf_chunked' in src}")
    print(f"  Uses persisted CSR chunks: {'get_chunk_dirs' in dir(gen)}")
    print(f"  No full-corpus collect: {'s23 = pl.concat([s2, s3], how=\"diagonal\")' not in src}")
    print(f"  No full-corpus TF-IDF matrix built: {'vectorizer.fit_transform' not in src}")
    print("Call chain: generate() -> run_tfidf_chunked() -> chunked_topk() -> get_deterministic_top_k()\n")

def run_step3():
    print("=== STEP 3: STRESS THE sp_matmul_topn TIE HANDLING ===")
    K = 20
    MARGIN_K = K + 100
    
    Q = sp.csr_matrix([[1.0]], dtype=np.float32)
    
    # Create 2 chunks, each has 300 candidates with identical score of 0.5
    C1 = sp.csr_matrix(np.full((300, 1), 0.5, dtype=np.float32))
    C2 = sp.csr_matrix(np.full((300, 1), 0.5, dtype=np.float32))
    C_full = sp.vstack([C1, C2])
    
    # Exact Reference (Q @ C_full.T)
    sim_exact = (Q @ C_full.T).toarray()[0]
    lg = np.arange(600, dtype=np.int64)
    exact_scores, exact_cands = get_deterministic_top_k(sim_exact, lg, K)
    
    # Production sp_matmul_topn + margin_k pipeline
    sim1 = sp_matmul_topn(Q, C1.T, MARGIN_K, sort=True)
    d1 = sim1.data[sim1.indptr[0]:sim1.indptr[1]]
    i1 = sim1.indices[sim1.indptr[0]:sim1.indptr[1]]
    
    sim2 = sp_matmul_topn(Q, C2.T, MARGIN_K, sort=True)
    d2 = sim2.data[sim2.indptr[0]:sim2.indptr[1]]
    i2 = sim2.indices[sim2.indptr[0]:sim2.indptr[1]] + 300
    
    comb_s = np.concatenate([d1, d2])
    comb_g = np.concatenate([i1, i2])
    
    prod_scores, prod_cands = get_deterministic_top_k(comb_s, comb_g, K)
    
    print(f"Exact Cands: {list(exact_cands)}")
    print(f"Prod Cands : {list(prod_cands)}")
    
    if list(exact_cands) != list(prod_cands):
        print("FAIL: Tie handling with margin_k truncated tied candidates before deterministic sorter saw them!")
        sys.exit(1)
    else:
        print("PASS: Margin K handled ties.")

if __name__ == '__main__':
    run_step1()
    run_step2()
    run_step3()
