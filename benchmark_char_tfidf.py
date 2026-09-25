import polars as pl
import numpy as np
import time
import os
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import scipy.sparse as sparse

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split

def run_benchmark():
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    s23 = pl.concat([frames["train_source2"].collect(), frames["train_source3"].collect()], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    t_split, v_split = get_grouped_split(gt['source1_entity_id'].to_list())
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(v_split, size=10000, replace=False).tolist())
    v_unseen = [x for x in v_split if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(v_unseen, size=20000, replace=False).tolist()
    
    # Take a small subset of 500 queries for benchmarking
    sub_ids = strict_val_ids[:500]
    s1_sub = s1.filter(pl.col("entity_id").is_in(sub_ids))
    
    # Frozen Config
    col = "latin_name"
    analyzer = "char_wb"
    ngram_range = (3,4)
    min_df = 2
    max_df = 0.1
    k = 20
    
    s1_texts = s1_sub[col].fill_null("").to_list()
    s23_texts = s23[col].fill_null("").to_list()
    
    print("Fitting TF-IDF on full corpus...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_df=max_df, dtype=np.float32)
    X_c = vectorizer.fit_transform(s23_texts)
    X_q = vectorizer.transform(s1_texts)
    print(f"Fit + Transform done in {time.time()-t0:.2f}s")
    
    vocab_size = len(vectorizer.vocabulary_)
    
    print(f"Query count: {X_q.shape[0]}")
    print(f"Corpus count: {X_c.shape[0]}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Average nonzeros/query: {X_q.nnz / X_q.shape[0]:.2f}")
    
    s1_ids = s1_sub["entity_id"].to_numpy()
    s23_ids = s23["entity_id"].to_numpy()
    
    # Old Method: SciPy Chunked Dot
    print("\n[OLD METHOD] SciPy Dot Product")
    t1 = time.time()
    X_c_T = X_c.T.tocsr()
    sim_old = X_q.dot(X_c_T)
    old_cands = {}
    for row_idx in range(sim_old.shape[0]):
        data = sim_old.data[sim_old.indptr[row_idx]:sim_old.indptr[row_idx+1]]
        indices = sim_old.indices[sim_old.indptr[row_idx]:sim_old.indptr[row_idx+1]]
        if len(data) > k:
            top_idx = np.argpartition(data, -k)[-k:]
            best = indices[top_idx]
            best_scores = data[top_idx]
        else:
            best = indices
            best_scores = data
        # sort by score descending for exact comparison
        sort_idx = np.argsort(-best_scores)
        old_cands[s1_ids[row_idx]] = (s23_ids[best[sort_idx]], best_scores[sort_idx])
    t_old = time.time() - t1
    print(f"SciPy Runtime: {t_old:.2f}s")
    
    del X_c_T, sim_old
    gc.collect()
    
    # New Method: sparse_dot_topn
    print("\n[NEW METHOD] sparse_dot_topn")
    t2 = time.time()
    X_c_csc = X_c.tocsc()
    sim_new = sp_matmul_topn(X_q, X_c_csc, top_n=k, sort=True)
    new_cands = {}
    for row_idx in range(sim_new.shape[0]):
        data = sim_new.data[sim_new.indptr[row_idx]:sim_new.indptr[row_idx+1]]
        indices = sim_new.indices[sim_new.indptr[row_idx]:sim_new.indptr[row_idx+1]]
        new_cands[s1_ids[row_idx]] = (s23_ids[indices], data)
    t_new = time.time() - t2
    print(f"sparse_dot_topn Runtime: {t_new:.2f}s")
    
    # Validate Equivalence
    print("\n[VALIDATION]")
    matches = 0
    score_diffs = []
    
    for sid in s1_ids:
        old_ids, old_s = old_cands[sid]
        new_ids, new_s = new_cands[sid]
        
        # Check sets of IDs (order might differ for exact ties)
        if set(old_ids) == set(new_ids):
            matches += 1
            
        # check score differences
        old_dict = {i: s for i, s in zip(old_ids, old_s)}
        new_dict = {i: s for i, s in zip(new_ids, new_s)}
        
        for i in set(old_ids) & set(new_ids):
            score_diffs.append(abs(old_dict[i] - new_dict[i]))
            
    print(f"Top-K Exact Set Matches: {matches} / {len(s1_ids)}")
    print(f"Max score diff on matched IDs: {np.max(score_diffs) if score_diffs else 0:.6e}")
    
if __name__ == "__main__":
    run_benchmark()
