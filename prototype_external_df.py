"""
prototype_external_df.py
-------------------------
Exact external-memory DF aggregation pipeline prototype.
- Replaces global in-memory Python DF dictionary with disk-backed k-way merge.
- Guarantees exact sklearn equivalent tokenization, vocabulary, and IDF.
- Implements deterministic tie-handling for top-K retrieval.
"""

import os, sys, json, time, shutil, heapq
import numpy as np
import polars as pl
import scipy.sparse as sp
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

PROC = psutil.Process(os.getpid())
def get_rss(): return PROC.memory_info().rss / (1024 * 1024)

BASE = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet"
SOURCE2 = BASE + "/train/source2.parquet"
SOURCE3 = BASE + "/train/source3.parquet"
OUTFILE = "external_df_output.txt"
_f = open(OUTFILE, "w", encoding="utf-8")

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    _f.write(msg + "\n")
    _f.flush()

CHANNELS = [
    {"name":"name_word",    "analyzer":"word",    "ngram_range":(1,1), "min_df":2, "max_df":0.01,  "col":"latin_name",    "K":50},
    {"name":"address_word", "analyzer":"word",    "ngram_range":(1,2), "min_df":2, "max_df":0.02,  "col":"latin_address", "K":20},
    {"name":"name_char",    "analyzer":"char_wb", "ngram_range":(3,4), "min_df":2, "max_df":0.005, "col":"latin_name",    "K":20},
]

# =========================================================================
# EXACT EXTERNAL-MEMORY DF AGGREGATION
# =========================================================================

def write_batch(local_df, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for term in sorted(local_df.keys()):
            f.write(f"{json.dumps(term)}\t{local_df[term]}\n")

def read_batch(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            t_str, c_str = line.rsplit('\t', 1)
            yield (json.loads(t_str), int(c_str))

def build_external_df(docs, analyzer_func, n_samples, min_df, max_df, batch_size, out_dir):
    """
    Builds vocabulary and IDF using external memory k-way merge.
    Returns: vocab, idf, stats_dict
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    rss_peak = get_rss()
    
    # 1. BATCHING PHASE
    batch_files = []
    local_df = {}
    docs_in_batch = 0
    batch_idx = 0
    
    for i, doc in enumerate(docs):
        for term in set(analyzer_func(doc)):
            local_df[term] = local_df.get(term, 0) + 1
        
        docs_in_batch += 1
        if docs_in_batch >= batch_size or i == len(docs) - 1:
            filepath = os.path.join(out_dir, f"batch_{batch_idx:04d}.tsv")
            write_batch(local_df, filepath)
            batch_files.append(filepath)
            local_df.clear()
            docs_in_batch = 0
            batch_idx += 1
            
            cur_rss = get_rss()
            if cur_rss > rss_peak: rss_peak = cur_rss

    disk_usage_batches = sum(os.path.getsize(f) for f in batch_files)

    # 2. MERGING PHASE
    generators = [read_batch(f) for f in batch_files]
    merged_stream = heapq.merge(*generators) # Tuples (term, count) sort correctly
    
    min_df_abs = min_df if isinstance(min_df, int) else int(np.ceil(min_df * n_samples))
    max_df_abs = max_df if isinstance(max_df, int) else int(np.floor(max_df * n_samples))
    
    vocab = {}
    df_counts_filtered = []
    raw_df_terms = 0
    
    current_term = None
    current_count = 0
    
    for term, count in merged_stream:
        if term == current_term:
            current_count += count
        else:
            if current_term is not None:
                raw_df_terms += 1
                if min_df_abs <= current_count <= max_df_abs:
                    vocab[current_term] = len(vocab)
                    df_counts_filtered.append(current_count)
            current_term = term
            current_count = count
            
    if current_term is not None:
        raw_df_terms += 1
        if min_df_abs <= current_count <= max_df_abs:
            vocab[current_term] = len(vocab)
            df_counts_filtered.append(current_count)
            
    cur_rss = get_rss()
    if cur_rss > rss_peak: rss_peak = cur_rss
    
    # 3. IDF CALCULATION
    df_arr = np.array(df_counts_filtered, dtype=np.float64)
    idf = (np.log((1.0 + n_samples) / (1.0 + df_arr)) + 1.0).astype(np.float32)
    
    vocab_disk_usage = len(json.dumps(vocab).encode('utf-8'))
    
    stats = {
        "batch_files": len(batch_files),
        "raw_df_terms": raw_df_terms,
        "filtered_vocab_size": len(vocab),
        "disk_usage_batches": disk_usage_batches,
        "vocab_disk_usage": vocab_disk_usage,
        "rss_peak": rss_peak
    }
    
    return vocab, idf, stats

# =========================================================================
# INCREMENTAL CSR TRANSFORM CHECK
# =========================================================================
def transform_docs_to_csr(docs, analyzer_func, vocab, idf):
    data_v, col_i, indptr = [], [], [0]
    for doc in docs:
        tc = {}
        for t in analyzer_func(doc):
            if t in vocab: tc[t] = tc.get(t, 0) + 1
        if tc:
            cols = [vocab[t] for t in tc]
            vals = np.array([tc[t] for t in tc], dtype=np.float32) * idf[cols]
            norm = np.linalg.norm(vals)
            if norm > 0: vals /= norm
            data_v.extend(vals.tolist())
            col_i.extend(cols)
        indptr.append(len(data_v))
    return sp.csr_matrix(
        (np.array(data_v, dtype=np.float32), np.array(col_i, dtype=np.int32), np.array(indptr, dtype=np.int32)),
        shape=(len(docs), len(vocab))
    )

# =========================================================================
# DETERMINISTIC TIE HANDLING RETRIEVAL (REGRESSION TEST)
# =========================================================================
def get_top_k_deterministic(scores, global_indices, K):
    """
    Permanent tie-handling fix for Top-K retrieval.
    Primary: descending score
    Secondary: ascending global row position
    """
    if len(scores) <= K:
        order = np.lexsort((global_indices, -scores))
        return scores[order], global_indices[order]
        
    kth_neg_score = np.partition(-scores, K-1)[K-1]
    boundary_score = -kth_neg_score
    
    candidates = np.where(scores >= boundary_score)[0]
    order = np.lexsort((global_indices[candidates], -scores[candidates]))
    top_k_idx = candidates[order[:K]]
    return scores[top_k_idx], global_indices[top_k_idx]


# =========================================================================
# PIPELINE EXECUTION
# =========================================================================

def main():
    log("="*70)
    log("LOADING 2M S2+S3 TRAIN CORPUS")
    log("="*70)
    
    N_2M = 2_000_000
    lf2 = pl.scan_parquet(SOURCE2).select(["entity_id", "latin_name", "latin_address"])
    lf3 = pl.scan_parquet(SOURCE3).select(["entity_id", "latin_name", "latin_address"])
    df = pl.concat([lf2, lf3]).head(N_2M).collect()
    
    docs_name = df["latin_name"].to_list()
    docs_addr = df["latin_address"].to_list()
    log(f"Loaded {len(df)} rows. Current RSS: {get_rss():.1f} MiB\n")
    
    # -----------------------------------------------------------------
    # SECTION 1: EQUIVALENCE CHECK (100k)
    # -----------------------------------------------------------------
    log("="*70)
    log("SECTION 1: EXTERNAL DF EQUIVALENCE CHECK (100k rows)")
    log("="*70)
    
    N_100K = 100_000
    for cfg in CHANNELS:
        log(f"\n--- Channel: {cfg['name']} ---")
        docs_100k = (docs_name if cfg["col"] == "latin_name" else docs_addr)[:N_100K]
        
        # In-memory Reference
        ref_vec = TfidfVectorizer(
            analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32
        )
        X_ref = ref_vec.fit_transform(docs_100k)
        ref_vocab = ref_vec.vocabulary_
        ref_idf = ref_vec.idf_.astype(np.float32)
        analyzer = ref_vec.build_analyzer()
        
        # External DF Build
        ext_vocab, ext_idf, stats = build_external_df(
            docs_100k, analyzer, N_100K, cfg["min_df"], cfg["max_df"], 
            batch_size=25_000, out_dir="tmp_ext_df"
        )
        
        # Verify
        vocab_keys_match = set(ref_vocab.keys()) == set(ext_vocab.keys())
        idx_match = all(ref_vocab[k] == ext_vocab[k] for k in ref_vocab)
        idf_diff = np.abs(ref_idf - ext_idf)
        idf_max_diff = float(np.max(idf_diff))
        
        X_ext = transform_docs_to_csr(docs_100k[:5000], analyzer, ext_vocab, ext_idf)
        X_ref_small = X_ref[:5000]
        mat_diff = float(np.max(np.abs((X_ref_small - X_ext).data))) if (X_ref_small - X_ext).data.size > 0 else 0.0
        
        log(f"  Vocab term set identical   : {'PASS' if vocab_keys_match else 'FAIL'}")
        log(f"  Index mapping identical    : {'PASS' if idx_match else 'FAIL'}")
        log(f"  Missing/Extra terms        : {len(set(ref_vocab) - set(ext_vocab))} / {len(set(ext_vocab) - set(ref_vocab))}")
        log(f"  IDF max diff               : {idf_max_diff:.2e} (<= 1e-6: {'PASS' if idf_max_diff <= 1e-6 else 'FAIL'})")
        log(f"  CSR matrix max diff (5k)   : {mat_diff:.2e} (<= 1e-6: {'PASS' if mat_diff <= 1e-6 else 'FAIL'})")

    # -----------------------------------------------------------------
    # SECTION 2: 2M EXTERNAL DF BENCHMARK
    # -----------------------------------------------------------------
    log("\n" + "="*70)
    log("SECTION 2: 2M EXTERNAL DF BENCHMARK")
    log("="*70)
    
    BATCH_SIZE_2M = 200_000
    
    for cfg in CHANNELS:
        log(f"\n--- Channel: {cfg['name']} (2M rows) ---")
        docs_2m = docs_name if cfg["col"] == "latin_name" else docs_addr
        
        ref_vec = TfidfVectorizer(
            analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32
        )
        ref_vec.fit(docs_2m[:1000]) # just to build analyzer
        analyzer = ref_vec.build_analyzer()
        
        t0 = time.time()
        vocab, idf, stats = build_external_df(
            docs_2m, analyzer, N_2M, cfg["min_df"], cfg["max_df"], 
            batch_size=BATCH_SIZE_2M, out_dir="tmp_ext_df"
        )
        t_elapsed = time.time() - t0
        
        log(f"  Total input rows           : {N_2M}")
        log(f"  Number of batch files      : {stats['batch_files']}")
        log(f"  Raw DF term count          : {stats['raw_df_terms']}")
        log(f"  Filtered vocabulary size   : {stats['filtered_vocab_size']}")
        log(f"  Peak process RSS           : {stats['rss_peak']:.1f} MiB")
        log(f"  DF runtime (s)             : {t_elapsed:.2f}")
        log(f"  Batch files disk usage     : {stats['disk_usage_batches'] / (1024*1024):.2f} MiB")
        log(f"  Final vocab disk usage     : {stats['vocab_disk_usage'] / (1024*1024):.2f} MiB")
        log(f"  Global Python DF Map used? : NO (Only local batches and external merge)")

    # -----------------------------------------------------------------
    # SECTION 3: TIE-HANDLING REGRESSION TEST
    # -----------------------------------------------------------------
    log("\n" + "="*70)
    log("SECTION 3: TIE-HANDLING REGRESSION TEST")
    log("="*70)
    
    # Generate mock tied scores
    np.random.seed(42)
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6])
    global_indices = np.array([100, 50, 80, 20, 90, 10, 5])
    K = 3
    
    log("  Mock data with ties at K-boundary:")
    log(f"  Scores : {scores}")
    log(f"  Indices: {global_indices}")
    log(f"  K = {K}")
    
    top_scores, top_idx = get_top_k_deterministic(scores, global_indices, K)
    
    log(f"\n  Deterministic Result (Primary: desc score, Secondary: asc index):")
    log(f"  Top Scores : {top_scores}")
    log(f"  Top Indices: {top_idx}")
    
    # Validation
    expected_idx = [100, 20, 50]
    expected_scores = [0.9, 0.8, 0.8]
    if list(top_idx) == expected_idx and list(top_scores) == expected_scores:
        log("  -> Tie-handling regression test: PASS")
    else:
        log(f"  -> Tie-handling regression test: FAIL (Expected idx {expected_idx}, got {list(top_idx)})")

    log("\n" + "="*70)
    log("PROTOTYPE COMPLETE")
    log("="*70)

if __name__ == "__main__":
    main()
