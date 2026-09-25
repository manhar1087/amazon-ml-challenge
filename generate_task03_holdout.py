import os
import time
import json
import polars as pl
import numpy as np
import scipy.sparse as sp
import gc
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from src.blocking.cross_script import cross_script_name_blocking
from src.blocking.abbreviation import abbreviation_blocking
from src.blocking.union import union_candidates
from src.blocking.sparse_retrieval import get_deterministic_top_k

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

def get_chunk_dirs(channel_name, chunk_size, chunk_dir="incremental_csr_output"):
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
                
                local_k = min(K, n_chunk)
                lg = np.arange(n_chunk, dtype=np.int64) + g_start
                
                comb_s = np.concatenate([best_scores[q_global], sim_row])
                comb_g = np.concatenate([best_gidx[q_global],   lg])
                
                bs, bg = get_deterministic_top_k(comb_s, comb_g, K)
                
                best_scores[q_global, :len(bs)] = bs
                best_gidx[q_global, :len(bg)]   = bg
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

def run_tfidf_chunked(s1, col, analyzer, ngram, k, channel_name, corpus_eids, setup_dir="retrieval_setup"):
    print(f"[{channel_name}] Loading precomputed vocab and IDF...")
    channel_map = {
        "tfidf_name_k50": "name_word",
        "tfidf_addr_k20": "address_word",
        "tfidf_char_k20": "name_char"
    }
    base_channel = channel_map[channel_name]
    sdir = os.path.join(setup_dir, base_channel)
    
    with open(os.path.join(sdir, "vocab.json")) as f:
        vocab = json.load(f)
    idf = np.load(os.path.join(sdir, "idf.npy"))
    
    # We just need the analyzer function from a dummy vectorizer
    vec_dummy = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram)
    analyzer_func = vec_dummy.build_analyzer()
    
    s1_texts = s1[col].fill_null("").to_list()
    print(f"[{channel_name}] Transforming S1 Queries...")
    X_q = transform_docs(s1_texts, vocab, idf, analyzer_func)
    
    vocab_size = len(vocab)
    print(f"[{channel_name}] Vocab size: {vocab_size}")
    
    chunk_dirs = get_chunk_dirs(base_channel, chunk_size=50000)
    print(f"[{channel_name}] Executing chunked Top-{k} retrieval over {len(chunk_dirs)} chunks...")
    
    out_s1, out_s23 = chunked_topk(X_q, chunk_dirs, vocab_size, k, s1["entity_id"].to_list(), corpus_eids)
    
    df = pl.DataFrame({
        "source1_entity_id": out_s1,
        "candidate_entity_id": out_s23,
        "retrieval_channels": [channel_name] * len(out_s1)
    })
    
    del X_q
    gc.collect()
    return df, vocab_size


def generate():
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    
    # DO NOT collect full corpus into memory!
    s2_lf = frames["train_source2"].lazy()
    s3_lf = frames["train_source3"].lazy()
    s23_lf = pl.concat([s2_lf, s3_lf], how="diagonal")
    
    print("Extracting 20k Holdout...")
    gt = frames["train_ground_truth"].collect()
    t_split, v_split = get_grouped_split(gt['source1_entity_id'].to_list())
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(v_split, size=10000, replace=False).tolist())
    v_unseen = [x for x in v_split if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(v_unseen, size=20000, replace=False).tolist()
    
    s1_20k = s1.filter(pl.col("entity_id").is_in(strict_val_ids))
    
    print("Executing Exact Blocks...")
    c_name = exact_name_blocking(s1_20k.lazy(), s23_lf)
    c_addr = exact_address_blocking(s1_20k.lazy(), s23_lf)
    c_ph = postal_house_blocking(s1_20k.lazy(), s23_lf)
    c_cs = cross_script_name_blocking(s1_20k.lazy(), s23_lf)
    c_abbr = abbreviation_blocking(s1_20k.lazy(), s23_lf)
    
    # We only need the entity IDs array for the corpus mapping
    print("Loading Corpus Entity IDs...")
    corpus_eids = frames["train_source2"].select("entity_id").collect().to_series().to_list() + frames["train_source3"].select("entity_id").collect().to_series().to_list()
    
    print("Executing TF-IDF Name Word (K=50)...")
    c_tfidf_name, v_wname = run_tfidf_chunked(s1_20k, "latin_name", "word", (1,1), 50, "tfidf_name_k50", corpus_eids)
    
    print("Executing TF-IDF Address Word (K=20)...")
    c_tfidf_addr, v_waddr = run_tfidf_chunked(s1_20k, "latin_address", "word", (1,2), 20, "tfidf_addr_k20", corpus_eids)
    
    print("Executing TF-IDF Name Char (K=20)...")
    c_tfidf_char, v_char = run_tfidf_chunked(s1_20k, "latin_name", "char_wb", (3,4), 20, "tfidf_char_k20", corpus_eids)
    
    print("Unioning Candidates...")
    all_cands = union_candidates([c_name, c_addr, c_ph, c_cs, c_abbr, c_tfidf_name, c_tfidf_addr, c_tfidf_char])
    
    # Ensure every holdout S1 has a row (even if empty) by left joining
    base_s1 = pl.DataFrame({"source1_entity_id": strict_val_ids})
    all_cands = base_s1.join(all_cands, on="source1_entity_id", how="left")
    
    os.makedirs("work/candidates", exist_ok=True)
    out_path = "work/candidates/task03_frozen_holdout_20k.parquet"
    all_cands.write_parquet(out_path)
    print(f"Saved exact holdout candidate artifact to {out_path}")
    
    # Get total heights
    s2_count = frames["train_source2"].select(pl.count()).collect()[0, 0]
    s3_count = frames["train_source3"].select(pl.count()).collect()[0, 0]
    
    manifest = {
        "holdout_s1_count": 20000,
        "s2_count": s2_count,
        "s3_count": s3_count,
        "k_values": {"tfidf_name": 50, "tfidf_addr": 20, "tfidf_char": 20},
        "vectorizer_config": "Frozen Vocabulary (min_df=2, max_df=bounds)",
        "chunk_size": 50000,
        "vocabulary_sizes": {"word_name": v_wname, "word_addr": v_waddr, "char_name": v_char},
        "generation_timestamp": datetime.now().isoformat(),
        "total_candidate_pairs": all_cands.height
    }
    with open("work/candidates/task03_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
if __name__ == "__main__":
    generate()

