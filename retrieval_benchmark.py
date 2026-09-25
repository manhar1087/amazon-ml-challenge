from src.blocking.sparse_retrieval import get_deterministic_top_k
"""
retrieval_benchmark.py
-----------------------
Retrieval stress benchmark using the validated incremental CSR representation.

Sections:
  0. Setup: recompute vocab+IDF for 1M corpus, save to disk
  1. Load & transform S1 queries (up to 20k)
  2. Chunked global top-K retrieval — all 3 channels
  3. Correctness verification vs monolithic reference
  4. address_word DF scaling at 2M rows
"""

import os, sys, json, time, hashlib
import numpy as np
import polars as pl
import scipy.sparse as sp
import psutil

PROC = psutil.Process(os.getpid())
def rss(): return PROC.memory_info().rss / (1024 * 1024)

BASE     = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet"
SOURCE1  = BASE + "/train/source1.parquet"
SOURCE2  = BASE + "/train/source2.parquet"
SOURCE3  = BASE + "/train/source3.parquet"
CHUNK_DIR = "incremental_csr_output"   # written by incremental_tfidf.py
CHUNK_SIZE = 50_000                     # chunk size used for CSR files
N_1M     = 1_000_000
N_QUERIES = 20_000                      # target; report if we use fewer
SETUP_DIR = "retrieval_setup"           # vocab/IDF saved here

OUTFILE = "retrieval_benchmark_output.txt"
_f = open(OUTFILE, "w", encoding="utf-8")
def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); _f.write(msg+"\n"); _f.flush()

CHANNELS = [
    {"name":"name_word",    "analyzer":"word",    "ngram_range":(1,1), "min_df":2, "max_df":0.01,  "col":"latin_name",    "K":50},
    {"name":"address_word", "analyzer":"word",    "ngram_range":(1,2), "min_df":2, "max_df":0.02,  "col":"latin_address", "K":20},
    {"name":"name_char",    "analyzer":"char_wb", "ngram_range":(3,4), "min_df":2, "max_df":0.005, "col":"latin_name",    "K":20},
]

# =========================================================================
# SECTION 0: Setup — vocab + IDF for 1M corpus
# =========================================================================
log("="*70)
log("SECTION 0 — Vocab/IDF setup for 1M corpus")
log("="*70)

from sklearn.feature_extraction.text import TfidfVectorizer

def alphabetical_vocab(docs, analyzer_func, n_samples, min_df, max_df):
    df_counts = {}
    for doc in docs:
        for t in set(analyzer_func(doc)):
            df_counts[t] = df_counts.get(t, 0) + 1
    min_abs = min_df if isinstance(min_df, int) else int(np.ceil(min_df * n_samples))
    max_abs = max_df if isinstance(max_df, int) else int(np.floor(max_df * n_samples))
    filtered = sorted(t for t,df in df_counts.items() if min_abs <= df <= max_abs)
    vocab = {t:i for i,t in enumerate(filtered)}
    idf_arr = np.array([df_counts[t] for t in filtered], dtype=np.float64)
    idf = (np.log((1+n_samples)/(1+idf_arr))+1.0).astype(np.float32)
    return vocab, idf, df_counts

log("Loading 1M S2+S3 corpus ...")
lf2 = pl.scan_parquet(SOURCE2).select(["entity_id","latin_name","latin_address"])
lf3 = pl.scan_parquet(SOURCE3).select(["entity_id","latin_name","latin_address"])
corpus_df = pl.concat([lf2, lf3]).head(N_1M).collect()
assert len(corpus_df) == N_1M
corpus_eids = corpus_df["entity_id"].to_list()
corpus_name = corpus_df["latin_name"].to_list()
corpus_addr = corpus_df["latin_address"].to_list()
log(f"  Corpus loaded: {len(corpus_df)} rows, RSS={rss():.0f} MiB")

os.makedirs(SETUP_DIR, exist_ok=True)
setup_meta = {}
for cfg in CHANNELS:
    sdir = os.path.join(SETUP_DIR, cfg["name"])
    vpath = sdir + "/vocab.json"
    ipath = sdir + "/idf.npy"
    if os.path.exists(vpath) and os.path.exists(ipath):
        log(f"  {cfg['name']}: loading saved vocab+IDF")
        with open(vpath) as f: vocab = json.load(f)
        idf = np.load(ipath)
    else:
        os.makedirs(sdir, exist_ok=True)
        log(f"  {cfg['name']}: building vocab+IDF ...")
        docs = corpus_name if cfg["col"]=="latin_name" else corpus_addr
        vec = TfidfVectorizer(analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
                              min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32)
        vec.fit(docs)
        analyzer_func = vec.build_analyzer()
        vocab, idf, _ = alphabetical_vocab(docs, analyzer_func, N_1M, cfg["min_df"], cfg["max_df"])
        with open(vpath, "w") as f: json.dump(vocab, f)
        np.save(ipath, idf)
        log(f"    vocab size={len(vocab)}, saved")
    setup_meta[cfg["name"]] = {"vocab": vocab, "idf": idf}
log()

# =========================================================================
# SECTION 1: Load & transform S1 queries
# =========================================================================
log("="*70)
log("SECTION 1 — Load & transform S1 queries")
log("="*70)

s1_df = pl.scan_parquet(SOURCE1).select(["entity_id","latin_name","latin_address"]).head(N_QUERIES).collect()
actual_queries = len(s1_df)
if actual_queries < N_QUERIES:
    log(f"  [NOTE] Only {actual_queries} S1 rows available (requested {N_QUERIES})")
else:
    log(f"  Loaded {actual_queries} S1 queries (target {N_QUERIES} met)")
q_eids = s1_df["entity_id"].to_list()
q_name = s1_df["latin_name"].to_list()
q_addr = s1_df["latin_address"].to_list()
log(f"  S1 RSS after load: {rss():.0f} MiB")

def transform_docs(docs, vocab, idf):
    """Transform docs to normalised CSR using frozen vocab+IDF."""
    n = len(docs)
    n_feat = len(vocab)
    data_v, col_i, indptr = [], [], [0]
    # Build analyzer from a minimal TfidfVectorizer with same settings
    # (We pass the vocab/idf externally; analyzer is reconstructed per channel)
    for doc in docs:
        tc = {}
        for t in _cur_analyzer(doc):
            if t in vocab:
                tc[t] = tc.get(t,0)+1
        if tc:
            cols = [vocab[t] for t in tc]
            vals = np.array([tc[t] for t in tc], dtype=np.float32) * idf[cols]
            norm = np.linalg.norm(vals)
            if norm > 0: vals /= norm
            data_v.extend(vals.tolist())
            col_i.extend(cols)
        indptr.append(len(data_v))
    return sp.csr_matrix(
        (np.array(data_v,dtype=np.float32), np.array(col_i,dtype=np.int32),
         np.array(indptr,dtype=np.int32)), shape=(n, n_feat))

query_matrices = {}
for cfg in CHANNELS:
    log(f"  Transforming S1 queries for {cfg['name']} ...")
    vocab = setup_meta[cfg["name"]]["vocab"]
    idf   = setup_meta[cfg["name"]]["idf"]
    docs  = q_name if cfg["col"]=="latin_name" else q_addr
    n_feat = len(vocab)
    # Build analyzer
    _vec_tmp = TfidfVectorizer(analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
                               min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32)
    _vec_tmp.fit(docs[:1000] if len(docs)>1000 else docs)
    global _cur_analyzer
    _cur_analyzer = _vec_tmp.build_analyzer()
    Q = transform_docs(docs, vocab, idf)
    log(f"    Q shape={Q.shape}, NNZ={Q.nnz}, RSS={rss():.0f} MiB")
    query_matrices[cfg["name"]] = Q
log()

# =========================================================================
# SECTION 2: Chunked global top-K retrieval
# =========================================================================
log("="*70)
log("SECTION 2 — Chunked global top-K retrieval (1M corpus)")
log("="*70)

QUERY_BATCH = 500   # process this many queries per chunk multiply to bound dense matrix size

def get_chunk_dirs(channel_name, chunk_size):
    p = os.path.join(CHUNK_DIR, channel_name, f"chunk_{chunk_size}")
    return sorted(
        os.path.join(p,d) for d in os.listdir(p)
        if os.path.isdir(os.path.join(p,d))
    )

def chunked_topk(Q, chunk_dirs, n_features, K, corpus_entity_ids):
    n_q = Q.shape[0]
    best_scores  = np.full((n_q, K), -np.inf, dtype=np.float64)
    best_gidx    = np.full((n_q, K), -1,      dtype=np.int64)

    import gc
    rss_peak = rss()
    for chunk_path in chunk_dirs:
        with open(chunk_path+"/metadata.json") as f: meta = json.load(f)
        g_start   = meta["global_row_start"]
        n_chunk   = meta["n_rows"]
        
        import scipy.sparse as sp
        C = sp.csr_matrix(
            (np.load(chunk_path+"/data.npy"),
             np.load(chunk_path+"/indices.npy"),
             np.load(chunk_path+"/indptr.npy")),
            shape=(n_chunk, n_features))

        for q0 in range(0, n_q, QUERY_BATCH):
            q1    = min(q0+QUERY_BATCH, n_q)
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
                
        rss_peak = max(rss_peak, rss())
        del C
        gc.collect()

    rows, cols, scores_out, gidx_out = [], [], [], []
    for q in range(n_q):
        valid = best_gidx[q] >= 0
        s = best_scores[q][valid]
        g = best_gidx[q][valid]
        if len(s) == 0: continue
        bs, bg = get_deterministic_top_k(s, g, K)
        for rank_i, (si, gi) in enumerate(zip(bs, bg)):
            rows.append(q)
            cols.append(rank_i)
            scores_out.append(float(si))
            gidx_out.append(int(gi))
            
    return rows, cols, scores_out, gidx_out, rss_peak

retrieval_results = {}
for cfg in CHANNELS:
    log(f"--- Channel: {cfg['name']}  K={cfg['K']} ---")
    Q       = query_matrices[cfg["name"]]
    vocab   = setup_meta[cfg["name"]]["vocab"]
    n_feat  = len(vocab)
    chunk_dirs = get_chunk_dirs(cfg["name"], CHUNK_SIZE)
    K = cfg["K"]

    rss_before = rss()
    t0 = time.time()
    rows, cols, scores, gidx, rss_peak = chunked_topk(Q, chunk_dirs, n_feat, K, corpus_eids)
    elapsed = time.time()-t0

    n_out = len(rows)
    queries_with_results = len(set(rows))
    queries_empty = actual_queries - queries_with_results

    log(f"  corpus rows        : {N_1M}")
    log(f"  query rows         : {actual_queries}")
    log(f"  n_corpus_chunks    : {len(chunk_dirs)}")
    log(f"  chunk_size         : {CHUNK_SIZE}")
    log(f"  K                  : {K}")
    log(f"  total output rows  : {n_out}")
    log(f"  queries with results: {queries_with_results}")
    log(f"  empty queries      : {queries_empty}")
    log(f"  avg candidates/q   : {n_out/actual_queries:.2f}")
    log(f"  total runtime (s)  : {elapsed:.2f}")
    log(f"  peak RSS (MiB)     : {rss_peak:.1f}")
    log(f"  RSS before (MiB)   : {rss_before:.1f}")
    retrieval_results[cfg["name"]] = {
        "rows":rows,"cols":cols,"scores":scores,"gidx":gidx,
        "elapsed":elapsed,"rss_peak":rss_peak
    }
    log()

# =========================================================================
# SECTION 3: Correctness verification
# =========================================================================
log("="*70)
log("SECTION 3 — Correctness verification (chunked vs monolithic)")
log("="*70)

N_VERIFY_Q  = 100    # first 100 queries
N_VERIFY_C  = 3      # first 3 chunks (150k corpus rows) → cross-chunk boundary included

log(f"  Verification subset: {N_VERIFY_Q} queries vs first {N_VERIFY_C} chunks")

for cfg in CHANNELS:
    log(f"  Channel: {cfg['name']}  K={cfg['K']}")
    vocab = setup_meta[cfg["name"]]["vocab"]
    idf   = setup_meta[cfg["name"]]["idf"]
    n_feat = len(vocab)
    K = cfg["K"]
    Q_small = query_matrices[cfg["name"]][:N_VERIFY_Q]

    # Build monolithic reference from first N_VERIFY_C chunks
    chunk_dirs = get_chunk_dirs(cfg["name"], CHUNK_SIZE)[:N_VERIFY_C]
    blocks = []
    for cp in chunk_dirs:
        with open(cp+"/metadata.json") as f: m = json.load(f)
        blocks.append(sp.csr_matrix(
            (np.load(cp+"/data.npy"),
             np.load(cp+"/indices.npy"),
             np.load(cp+"/indptr.npy")),
            shape=(m["n_rows"], n_feat)))
    C_ref = sp.vstack(blocks, format="csr")    # monolithic (3×chunk_size rows)
    n_ref_rows = C_ref.shape[0]
    log(f"    monolithic ref shape: {C_ref.shape}")

    # Monolithic reference top-K
    sim_mono = (Q_small @ C_ref.T).toarray()   # (N_VERIFY_Q × n_ref_rows)
    ref_topk_idx   = np.zeros((N_VERIFY_Q, K), dtype=np.int64)
    ref_topk_score = np.zeros((N_VERIFY_Q, K), dtype=np.float64)
    for q in range(N_VERIFY_Q):
        local_k = min(K, n_ref_rows)
        li = np.argpartition(-sim_mono[q], local_k-1)[:local_k]
        ls = sim_mono[q][li]
        order = np.lexsort((li, -ls))
        li_all = np.arange(n_ref_rows, dtype=np.int64)
        bs, bi = get_deterministic_top_k(sim_mono[q], li_all, K)
        ref_topk_idx[q][:len(bi)] = bi
        ref_topk_score[q][:len(bs)] = bs

    # Chunked result for same N_VERIFY_Q queries vs same N_VERIFY_C chunks
    chunk_dirs_all = get_chunk_dirs(cfg["name"], CHUNK_SIZE)
    rows_v, cols_v, scores_v, gidx_v, _ = chunked_topk(
        Q_small, chunk_dirs_all[:N_VERIFY_C], n_feat, K, corpus_eids)

    # Reorganise chunked output
    ch_topk_idx   = np.full((N_VERIFY_Q, K), -1,      dtype=np.int64)
    ch_topk_score = np.full((N_VERIFY_Q, K), -np.inf, dtype=np.float64)
    for r,c,s,g in zip(rows_v, cols_v, scores_v, gidx_v):
        ch_topk_idx[r,c]   = g
        ch_topk_score[r,c] = s

    # Compare
    id_match    = np.all(ref_topk_idx == ch_topk_idx)
    score_diff  = np.abs(ref_topk_score - ch_topk_score)
    max_sdiff   = float(np.max(score_diff[ref_topk_idx>=0]))
    exceed_tol  = int(np.sum(score_diff[ref_topk_idx>=0] > 1e-7))

    log(f"    candidate ID match  : {'PASS' if id_match else 'FAIL'}")
    log(f"    max score diff      : {max_sdiff:.2e}")
    log(f"    score diffs > 1e-7  : {exceed_tol}")
    log(f"    ordering match      : {'PASS' if id_match else 'FAIL'}")

    # Cross-chunk boundary case: find a query whose top candidates span chunks
    for q in range(N_VERIFY_Q):
        top_gidx = ch_topk_idx[q]
        top_gidx = top_gidx[top_gidx >= 0]
        if len(top_gidx) >= 2:
            chunks_seen = set(int(g)//CHUNK_SIZE for g in top_gidx)
            if len(chunks_seen) > 1:
                log(f"    cross-chunk boundary: query {q} has candidates in chunks {sorted(chunks_seen)} -> verified above")
                break
    log()

# =========================================================================
# SECTION 4: address_word DF scaling at 2M rows
# =========================================================================
log("="*70)
log("SECTION 4 — address_word DF scaling at 2M rows")
log("="*70)

N_2M = 2_000_000
log(f"Loading {N_2M} rows from S2+S3 train corpus ...")
rss_pre = rss()
df_2m = pl.concat([
    pl.scan_parquet(SOURCE2).select(["entity_id","latin_address"]),
    pl.scan_parquet(SOURCE3).select(["entity_id","latin_address"]),
]).head(N_2M).collect()
rss_post_load = rss()
docs_2m = df_2m["latin_address"].to_list()
log(f"  RSS before load : {rss_pre:.0f} MiB")
log(f"  RSS after load  : {rss_post_load:.0f} MiB")

cfg_aw = CHANNELS[1]  # address_word
vec2m = TfidfVectorizer(analyzer=cfg_aw["analyzer"], ngram_range=cfg_aw["ngram_range"],
                         min_df=cfg_aw["min_df"], max_df=cfg_aw["max_df"], dtype=np.float32)
vec2m.fit(docs_2m[:1000])
analyzer_2m = vec2m.build_analyzer()

t0 = time.time()
df_counts_2m = {}
rss_peak_df = rss_post_load
DF_SUB = 100_000
for i in range(0, N_2M, DF_SUB):
    for doc in docs_2m[i:i+DF_SUB]:
        for t in set(analyzer_2m(doc)):
            df_counts_2m[t] = df_counts_2m.get(t,0)+1
    now = rss(); 
    if now > rss_peak_df: rss_peak_df = now

min_abs = cfg_aw["min_df"]
max_abs_f = int(np.floor(cfg_aw["max_df"] * N_2M))
filtered_2m = sorted(t for t,df in df_counts_2m.items() if min_abs <= df <= max_abs_f)
vocab_2m = {t:i for i,t in enumerate(filtered_2m)}
rss_post_vocab = rss()
elapsed_2m = time.time()-t0

log(f"  RSS peak during DF pass : {rss_peak_df:.1f} MiB")
log(f"  RSS after vocab build   : {rss_post_vocab:.1f} MiB")
log(f"  DF/vocab runtime (s)    : {elapsed_2m:.2f}")
log(f"  Raw DF dict size        : {len(df_counts_2m)} terms")
log(f"  Filtered vocab size     : {len(vocab_2m)}")
log(f"  (1M vocab was 717,963 — growth factor: {len(vocab_2m)/717963:.3f}x)")
log()

log("="*70)
log("RETRIEVAL BENCHMARK COMPLETE")
log(f"Output saved to: {OUTFILE}")
log("="*70)
_f.close()
