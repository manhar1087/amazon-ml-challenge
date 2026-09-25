import time
import os
import polars as pl
import numpy as np
import scipy.sparse as sp
import json
import gc

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.evaluation.candidate_recall import evaluate_candidates, print_recall_report
from src.blocking.sparse_retrieval import get_deterministic_top_k
from sparse_dot_topn import sp_matmul_topn

def get_chunk_dirs(channel_name, chunk_size=50000, chunk_dir="incremental_csr_output"):
    p = os.path.join(chunk_dir, channel_name, f"chunk_{chunk_size}")
    return sorted(
        os.path.join(p,d) for d in os.listdir(p)
        if os.path.isdir(os.path.join(p,d))
    )

def chunked_topk(Q, chunk_dirs, n_features, K, s1_ids, corpus_entity_ids, query_batch=500):
    n_q = Q.shape[0]
    best_scores  = np.full((n_q, K), -np.inf, dtype=np.float64)
    best_gidx    = np.full((n_q, K), -1,      dtype=np.int64)

    for chunk_path in chunk_dirs:
        with open(chunk_path+"/metadata.json") as f: meta = json.load(f)
        g_start   = meta["global_row_start"]
        n_chunk   = meta["n_rows"]
        
        C = sp.csr_matrix(
            (np.load(chunk_path+"/data.npy"),
             np.load(chunk_path+"/indices.npy"),
             np.load(chunk_path+"/indptr.npy")),
            shape=(n_chunk, n_features))

        for q0 in range(0, n_q, query_batch):
            q1    = min(q0+query_batch, n_q)
            Qb    = Q[q0:q1]
            sim   = (Qb @ C.T).toarray() 
            
            for qi in range(q1-q0):
                q_global = q0+qi
                sim_row  = sim[qi].astype(np.float64)
                lg = np.arange(n_chunk, dtype=np.int64) + g_start
                
                comb_s = np.concatenate([best_scores[q_global], sim_row])
                comb_g = np.concatenate([best_gidx[q_global], lg])
                valid = comb_g >= 0
                
                bs, bg = get_deterministic_top_k(comb_s[valid], comb_g[valid], K)
                
                best_scores[q_global, :len(bs)] = bs
                best_gidx[q_global, :len(bg)] = bg
                
        del C
        gc.collect()

    out_s1 = []
    out_s23 = []
    for q in range(n_q):
        valid = best_gidx[q] >= 0
        s = best_scores[q][valid]
        g = best_gidx[q][valid]
        if len(s) == 0: continue
        bs, bg = get_deterministic_top_k(s, g, K)
        s1_id = s1_ids[q]
        out_s1.extend([s1_id] * len(bg))
        out_s23.extend([corpus_entity_ids[int(idx)] for idx in bg])
        
    return out_s1, out_s23

def transform_docs(docs, vocab, idf, analyzer_func):
    n = len(docs)
    n_feat = len(vocab)
    dv, ci, ip = [], [], [0]
    for doc in docs:
        tc = {}
        for t in analyzer_func(doc):
            if t in vocab: tc[t] = tc.get(t, 0) + 1
        if tc:
            cols = [vocab[t] for t in tc]
            vals = np.array([tc[t] for t in tc], dtype=np.float32) * idf[cols]
            norm = np.linalg.norm(vals)
            if norm > 0: vals /= norm
            dv.extend(vals.tolist())
            ci.extend(cols)
        ip.append(len(dv))
    return sp.csr_matrix((np.array(dv, dtype=np.float32),
                          np.array(ci, dtype=np.int32),
                          np.array(ip, dtype=np.int32)), shape=(n, n_feat))

def run_test():
    print("Loading data...")
    frames = load_all_data()
    gt_df = frames["train_ground_truth"].collect()
    
    # 10k deterministic validation queries
    gt_dict = {}
    for row in gt_df.iter_rows():
        s1 = row[0]
        s2 = row[1]
        if s1 not in gt_dict: gt_dict[s1] = []
        gt_dict[s1].append(s2)
        
    t_split, v_split = get_grouped_split(gt_df['source1_entity_id'].to_list())
    np.random.seed(999)
    val_ids = np.random.choice(v_split, size=10000, replace=False).tolist()
    
    s1_all = frames["train_source1"].collect()
    s1 = s1_all.filter(pl.col("entity_id").is_in(val_ids))
    s1_ids = s1["entity_id"].to_list()
    s1_texts = s1["latin_name"].fill_null("").to_list()
    print(f"S1 queries loaded: {len(s1)}")
    
    # Load 1M Corpus Entity IDs
    s2 = frames["train_source2"].lazy()
    s3 = frames["train_source3"].lazy()
    corpus = pl.concat([s2, s3], how="diagonal").head(1000000).collect()
    corpus_eids = corpus["entity_id"].to_list()
    corpus_texts = corpus["latin_name"].fill_null("").to_list()
    
    # Using name_word (K=50) for the recall comparison
    K = 50
    channel_name = "name_word"
    setup_dir = "retrieval_setup"
    
    with open(os.path.join(setup_dir, channel_name, "vocab.json")) as f:
        vocab = json.load(f)
    idf = np.load(os.path.join(setup_dir, channel_name, "idf.npy"))
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec_dummy = TfidfVectorizer(analyzer="word", ngram_range=(1,1))
    analyzer_func = vec_dummy.build_analyzer()
    
    print("Transforming S1 queries...")
    X_q = transform_docs(s1_texts, vocab, idf, analyzer_func)
    print("Transforming 1M Corpus...")
    X_c = transform_docs(corpus_texts, vocab, idf, analyzer_func)
    
    # A. OLD sp_matmul_topn candidate set
    print("Executing sp_matmul_topn (OLD)...")
    sim = sp_matmul_topn(X_q, X_c.T, K, sort=True)
    out_s1_old = []
    out_s23_old = []
    for row_idx in range(sim.shape[0]):
        row_indices = sim.indices[sim.indptr[row_idx]:sim.indptr[row_idx+1]]
        s1_id = s1_ids[row_idx]
        out_s1_old.extend([s1_id] * len(row_indices))
        out_s23_old.extend([corpus_eids[idx] for idx in row_indices])
    
    df_old = pl.DataFrame({
        "source1_entity_id": out_s1_old,
        "candidate_entity_id": out_s23_old
    })
    
    # B. NEW deterministic chunked candidate set
    print("Executing deterministic chunked (NEW)...")
    chunk_dirs = get_chunk_dirs(channel_name, chunk_size=50000)
    vocab_size = len(vocab)
    out_s1_new, out_s23_new = chunked_topk(X_q, chunk_dirs, vocab_size, K, s1_ids, corpus_eids)
    
    df_new = pl.DataFrame({
        "source1_entity_id": out_s1_new,
        "candidate_entity_id": out_s23_new
    })
    
    print("Evaluating OLD candidate set...")
    metrics_old = evaluate_candidates(df_old, gt_dict, val_ids)
    print_recall_report(metrics_old, "OLD sp_matmul_topn")
    
    print("Evaluating NEW candidate set...")
    metrics_new = evaluate_candidates(df_new, gt_dict, val_ids)
    print_recall_report(metrics_new, "NEW deterministic chunked")

if __name__ == '__main__':
    run_test()
