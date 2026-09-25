"""
verify_correctness_fix.py
--------------------------
Re-runs ONLY the correctness verification section with proper
deterministic tie-breaking:
  - primary:   descending score
  - secondary: ascending global corpus row index

Both the monolithic reference and the chunked merge use the same ordering.
"""

import os, json, time
import numpy as np
import polars as pl
import scipy.sparse as sp
import psutil

PROC = psutil.Process(os.getpid())
def rss(): return PROC.memory_info().rss/(1024*1024)

BASE     = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet"
SOURCE1  = BASE + "/train/source1.parquet"
SOURCE2  = BASE + "/train/source2.parquet"
SOURCE3  = BASE + "/train/source3.parquet"
CHUNK_DIR = "incremental_csr_output"
CHUNK_SIZE = 50_000
N_1M      = 1_000_000
SETUP_DIR  = "retrieval_setup"

OUTFILE = "correctness_fix_output.txt"
_f = open(OUTFILE, "w", encoding="utf-8")
def log(*a):
    msg=" ".join(str(x) for x in a)
    print(msg); _f.write(msg+"\n"); _f.flush()

CHANNELS = [
    {"name":"name_word",    "analyzer":"word",    "ngram_range":(1,1), "min_df":2, "max_df":0.01,  "col":"latin_name",    "K":50},
    {"name":"address_word", "analyzer":"word",    "ngram_range":(1,2), "min_df":2, "max_df":0.02,  "col":"latin_address", "K":20},
    {"name":"name_char",    "analyzer":"char_wb", "ngram_range":(3,4), "min_df":2, "max_df":0.005, "col":"latin_name",    "K":20},
]

# ---- Load saved vocab/IDF ----
from sklearn.feature_extraction.text import TfidfVectorizer

setup_meta = {}
for cfg in CHANNELS:
    sdir = os.path.join(SETUP_DIR, cfg["name"])
    with open(sdir+"/vocab.json") as f: vocab = json.load(f)
    idf = np.load(sdir+"/idf.npy")
    setup_meta[cfg["name"]] = {"vocab":vocab,"idf":idf}

# ---- Load S1 queries and transform ----
log("Loading S1 queries ...")
s1_df = pl.scan_parquet(SOURCE1).select(["entity_id","latin_name","latin_address"]).head(20000).collect()
q_name = s1_df["latin_name"].to_list()
q_addr = s1_df["latin_address"].to_list()

_cur_analyzer = None

def transform_docs(docs, vocab, idf):
    n=len(docs); n_feat=len(vocab)
    dv,ci,ip=[],[],[0]
    for doc in docs:
        tc={}
        for t in _cur_analyzer(doc):
            if t in vocab: tc[t]=tc.get(t,0)+1
        if tc:
            cols=[vocab[t] for t in tc]
            vals=np.array([tc[t] for t in tc],dtype=np.float32)*idf[cols]
            norm=np.linalg.norm(vals)
            if norm>0: vals/=norm
            dv.extend(vals.tolist()); ci.extend(cols)
        ip.append(len(dv))
    return sp.csr_matrix((np.array(dv,dtype=np.float32),
                          np.array(ci,dtype=np.int32),
                          np.array(ip,dtype=np.int32)),shape=(n,n_feat))

query_matrices = {}
for cfg in CHANNELS:
    vocab=setup_meta[cfg["name"]]["vocab"]; idf=setup_meta[cfg["name"]]["idf"]
    docs=q_name if cfg["col"]=="latin_name" else q_addr
    _vec_tmp=TfidfVectorizer(analyzer=cfg["analyzer"],ngram_range=cfg["ngram_range"],
                              min_df=cfg["min_df"],max_df=cfg["max_df"],dtype=np.float32)
    _vec_tmp.fit(docs[:500])
    _cur_analyzer=_vec_tmp.build_analyzer()
    query_matrices[cfg["name"]]=transform_docs(docs,vocab,idf)
log(f"Query matrices built. RSS={rss():.0f} MiB")

def get_chunk_dirs(channel_name, chunk_size, limit=None):
    p=os.path.join(CHUNK_DIR,channel_name,f"chunk_{chunk_size}")
    dirs=sorted(os.path.join(p,d) for d in os.listdir(p) if os.path.isdir(os.path.join(p,d)))
    return dirs[:limit] if limit else dirs

# =========================================================================
# DETERMINISTIC chunked top-K with proper tie-breaking
# =========================================================================
QUERY_BATCH = 500

def topk_deterministic(scores, global_indices, K):
    """
    Return the K best (score desc, global_idx asc) from scores/global_indices arrays.
    Fully deterministic: lexsort, not argpartition.
    """
    if len(scores) <= K:
        order = np.lexsort((global_indices, -scores))
        return scores[order], global_indices[order]
    # Partial sort: keep only entries that could be in the top K
    # Use argpartition to get rough top-K, then sort exactly
    part = np.argpartition(-scores, K-1)[:K]  # not sorted, but top-K candidates
    # Check for ties at the boundary — find the K-th score value
    kth_score = np.partition(-scores, K-1)[K-1]  # = -K_th_largest score
    # Include ALL elements with score >= -kth_score (= score <= kth_score -> score >= boundary)
    boundary_score = -kth_score  # actual K-th largest score
    candidates = np.where(scores >= boundary_score)[0]
    # Now sort deterministically among candidates
    order = np.lexsort((global_indices[candidates], -scores[candidates]))
    return scores[candidates[order[:K]]], global_indices[candidates[order[:K]]]

def chunked_topk_det(Q, chunk_dirs, n_features, K):
    """Chunked retrieval with deterministic tie-breaking."""
    n_q = Q.shape[0]
    best_scores = np.full((n_q, K), -np.inf, dtype=np.float64)
    best_gidx   = np.full((n_q, K), -1,      dtype=np.int64)

    for chunk_path in chunk_dirs:
        with open(chunk_path+"/metadata.json") as f: meta=json.load(f)
        g_start = meta["global_row_start"]
        n_chunk = meta["n_rows"]
        C=sp.csr_matrix(
            (np.load(chunk_path+"/data.npy"),
             np.load(chunk_path+"/indices.npy"),
             np.load(chunk_path+"/indptr.npy")),
            shape=(n_chunk,n_features))

        for q0 in range(0,n_q,QUERY_BATCH):
            q1=min(q0+QUERY_BATCH,n_q)
            sim=(Q[q0:q1]@C.T).toarray()
            for qi in range(q1-q0):
                q_g=q0+qi
                sim_row=sim[qi].astype(np.float64)
                local_k=min(K,n_chunk)
                lg=(np.arange(n_chunk,dtype=np.int64)+g_start)
                # Merge: combine existing best with local chunk scores
                comb_s=np.concatenate([best_scores[q_g], sim_row])
                comb_g=np.concatenate([best_gidx[q_g],   lg])
                # Deterministic topk
                bs,bg=topk_deterministic(comb_s,comb_g,K)
                best_scores[q_g,:len(bs)]=bs
                best_gidx[q_g,:len(bg)]=bg
        del C

    # Build output
    rows,cols,scores_out,gidx_out=[],[],[],[]
    for q in range(n_q):
        valid=best_gidx[q]>=0
        s=best_scores[q][valid]; g=best_gidx[q][valid]
        if len(s)==0: continue
        bs,bg=topk_deterministic(s,g,K)
        for rank_i,(si,gi) in enumerate(zip(bs,bg)):
            rows.append(q); cols.append(rank_i)
            scores_out.append(float(si)); gidx_out.append(int(gi))
    return rows,cols,scores_out,gidx_out

# =========================================================================
# CORRECTNESS VERIFICATION
# =========================================================================
log()
log("="*70)
log("CORRECTNESS VERIFICATION — deterministic tie-breaking")
log("="*70)

N_VERIFY_Q = 100
N_VERIFY_C = 3

log(f"  {N_VERIFY_Q} queries vs first {N_VERIFY_C} chunks (cross-chunk boundary included)")

for cfg in CHANNELS:
    log(f"\n  Channel: {cfg['name']}  K={cfg['K']}")
    vocab=setup_meta[cfg["name"]]["vocab"]; n_feat=len(vocab)
    K=cfg["K"]
    Q_small=query_matrices[cfg["name"]][:N_VERIFY_Q]

    # Monolithic reference — load first N_VERIFY_C chunks, compute all at once
    chunk_dirs_v=get_chunk_dirs(cfg["name"],CHUNK_SIZE)[:N_VERIFY_C]
    blocks=[]
    for cp in chunk_dirs_v:
        with open(cp+"/metadata.json") as f: m=json.load(f)
        blocks.append(sp.csr_matrix(
            (np.load(cp+"/data.npy"),
             np.load(cp+"/indices.npy"),
             np.load(cp+"/indptr.npy")),
            shape=(m["n_rows"],n_feat)))
    C_ref=sp.vstack(blocks,format="csr")
    n_ref=C_ref.shape[0]
    log(f"    monolithic shape: {C_ref.shape}")

    sim_mono=(Q_small@C_ref.T).toarray().astype(np.float64)
    ref_gidx  =np.full((N_VERIFY_Q,K),-1,     dtype=np.int64)
    ref_score =np.full((N_VERIFY_Q,K),-np.inf,dtype=np.float64)
    for q in range(N_VERIFY_Q):
        s=sim_mono[q]; g=np.arange(n_ref,dtype=np.int64)
        bs,bg=topk_deterministic(s,g,K)
        ref_score[q,:len(bs)]=bs; ref_gidx[q,:len(bg)]=bg

    # Chunked with deterministic merge — same N_VERIFY_C chunks
    rows_v,cols_v,scores_v,gidx_v=chunked_topk_det(Q_small,chunk_dirs_v,n_feat,K)
    ch_gidx  =np.full((N_VERIFY_Q,K),-1,     dtype=np.int64)
    ch_score =np.full((N_VERIFY_Q,K),-np.inf,dtype=np.float64)
    for r,c,s,g in zip(rows_v,cols_v,scores_v,gidx_v):
        ch_gidx[r,c]=g; ch_score[r,c]=s

    # Compare
    id_match   =np.all(ref_gidx==ch_gidx)
    score_diff =np.abs(ref_score-ch_score)
    valid_mask =(ref_gidx>=0)&(ch_gidx>=0)
    max_sdiff  =float(np.max(score_diff[valid_mask])) if valid_mask.any() else 0.0
    n_exceed   =int(np.sum(score_diff[valid_mask]>1e-7))

    log(f"    candidate ID match  : {'PASS' if id_match else 'FAIL'}")
    log(f"    max score diff      : {max_sdiff:.2e}")
    log(f"    score diffs > 1e-7  : {n_exceed}")
    log(f"    ordering match      : {'PASS' if id_match else 'FAIL'}")

    # Report any mismatches in detail
    if not id_match:
        mismatch_qs=[q for q in range(N_VERIFY_Q) if not np.all(ref_gidx[q]==ch_gidx[q])]
        log(f"    mismatched queries  : {len(mismatch_qs)} — example q={mismatch_qs[0]}")
        q=mismatch_qs[0]
        log(f"      ref  top-5 gidx  : {ref_gidx[q,:5].tolist()}")
        log(f"      ch   top-5 gidx  : {ch_gidx[q,:5].tolist()}")
        log(f"      ref  top-5 score : {[round(x,6) for x in ref_score[q,:5].tolist()]}")
        log(f"      ch   top-5 score : {[round(x,6) for x in ch_score[q,:5].tolist()]}")

    # Cross-chunk boundary case
    for q in range(N_VERIFY_Q):
        gidx_q=ch_gidx[q]; gidx_q=gidx_q[gidx_q>=0]
        if len(gidx_q)>=2:
            chunks_seen=set(int(g)//CHUNK_SIZE for g in gidx_q)
            if len(chunks_seen)>1:
                log(f"    cross-chunk boundary: query {q} candidates span chunks {sorted(chunks_seen)} -> PASS")
                break

log()
log("="*70)
log("VERIFICATION COMPLETE")
log(f"Output: {OUTFILE}")
log("="*70)
_f.close()
