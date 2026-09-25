"""
incremental_tfidf.py
---------------------
Production-grade streaming TF-IDF with:
  1. Alphabetically-sorted vocabulary (matching sklearn exactly)
  2. True per-chunk CSR writing (no global accumulation)
  3. Actual process RSS measurement via psutil

Sections:
  A. Vocabulary fix + 200k clean verification (index mismatches must = 0)
  B. Incremental CSR writer benchmark on 1M-row corpus
  C. Reconstruction verification
  D. DF/vocabulary memory profiling
"""

import os
import sys
import time
import json
import shutil
import hashlib
import numpy as np
import polars as pl
import psutil
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

PROC = psutil.Process(os.getpid())

def rss_mib():
    return PROC.memory_info().rss / (1024 * 1024)

BASE    = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet"
SOURCE2 = BASE + "/train/source2.parquet"
SOURCE3 = BASE + "/train/source3.parquet"

OUTFILE = "incremental_tfidf_output.txt"
_fout = open(OUTFILE, "w", encoding="utf-8")

def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    _fout.write(msg + "\n")
    _fout.flush()

CHANNELS = [
    {"name": "name_word",    "analyzer": "word",    "ngram_range": (1,1), "min_df": 2, "max_df": 0.01,  "col": "latin_name"},
    {"name": "address_word", "analyzer": "word",    "ngram_range": (1,2), "min_df": 2, "max_df": 0.02,  "col": "latin_address"},
    {"name": "name_char",    "analyzer": "char_wb", "ngram_range": (3,4), "min_df": 2, "max_df": 0.005, "col": "latin_name"},
]

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_corpus(n_rows, extra_cols=None):
    """Load first n_rows from S2+S3 train corpus deterministically."""
    cols = ["entity_id", "latin_name", "latin_address"]
    lf2 = pl.scan_parquet(SOURCE2).select(cols)
    lf3 = pl.scan_parquet(SOURCE3).select(cols)
    df  = pl.concat([lf2, lf3]).head(n_rows).collect()
    assert len(df) == n_rows
    return df

def corpus_hash(entity_ids):
    h = hashlib.sha256()
    for eid in entity_ids:
        h.update(str(eid).encode("utf-8"))
    return h.hexdigest()

def build_vocab_streaming(docs, analyzer_func, n_samples, min_df, max_df):
    """
    First pass: compute DF counts.
    FIXED: vocabulary is ALPHABETICALLY SORTED (matching sklearn).
    Returns: vocab dict {term: int_idx}, df_counts dict.
    """
    df_counts = {}
    for doc in docs:
        for term in set(analyzer_func(doc)):
            df_counts[term] = df_counts.get(term, 0) + 1

    min_df_abs = min_df if isinstance(min_df, int) else int(np.ceil(min_df * n_samples))
    max_df_abs = max_df if isinstance(max_df, int) else int(np.floor(max_df * n_samples))

    # ALPHABETICALLY SORTED – matches sklearn vocabulary_ exactly
    filtered = sorted(
        term for term, df in df_counts.items()
        if min_df_abs <= df <= max_df_abs
    )
    vocab = {term: idx for idx, term in enumerate(filtered)}
    return vocab, df_counts

def build_idf(vocab, df_counts, n_samples):
    """IDF using sklearn formula, float32."""
    df_arr = np.array([df_counts[t] for t in sorted(vocab, key=vocab.get)], dtype=np.float64)
    idf = (np.log((1.0 + n_samples) / (1.0 + df_arr)) + 1.0).astype(np.float32)
    return idf

def transform_docs_to_csr(docs, analyzer_func, vocab, idf):
    """Transform a list of docs to a normalised CSR matrix (local, in memory)."""
    data_vals, col_idx, indptr = [], [], [0]
    for doc in docs:
        term_counts = {}
        for term in analyzer_func(doc):
            if term in vocab:
                term_counts[term] = term_counts.get(term, 0) + 1
        if term_counts:
            cols = [vocab[t] for t in term_counts]
            tfs  = np.array([term_counts[t] for t in term_counts], dtype=np.float32)
            vals = tfs * idf[cols]
            norm = np.linalg.norm(vals)
            if norm > 0.0:
                vals = vals / norm
            data_vals.extend(vals.tolist())
            col_idx.extend(cols)
        indptr.append(len(data_vals))
    n_docs = len(docs)
    n_feats = len(vocab)
    X = sp.csr_matrix(
        (np.array(data_vals, dtype=np.float32),
         np.array(col_idx,   dtype=np.int32),
         np.array(indptr,    dtype=np.int32)),
        shape=(n_docs, n_feats),
    )
    return X


# =======================================================================
# SECTION A: BLOCKER 1 — vocabulary ordering fix + 200k verification
# =======================================================================
log("=" * 70)
log("SECTION A — Blocker 1: Vocabulary ordering fix + 200k verification")
log("=" * 70)

N_VERIFY = 200_000
df_verify = load_corpus(N_VERIFY)
log(f"Corpus rows loaded : {len(df_verify)}")
log(f"Corpus hash        : {corpus_hash(df_verify['entity_id'].to_list())}")
log()

for cfg in CHANNELS:
    log(f"--- Channel: {cfg['name']} ---")
    docs = df_verify[cfg["col"]].to_list()
    n    = len(docs)

    # Reference sklearn
    ref_vec = TfidfVectorizer(
        analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32,
    )
    X_ref      = ref_vec.fit_transform(docs)
    ref_vocab  = ref_vec.vocabulary_
    ref_idf    = ref_vec.idf_.astype(np.float32)

    # Streaming with ALPHABETICALLY SORTED vocabulary
    analyzer_func = ref_vec.build_analyzer()
    stream_vocab, df_counts = build_vocab_streaming(
        docs, analyzer_func, n, cfg["min_df"], cfg["max_df"]
    )
    stream_idf = build_idf(stream_vocab, df_counts, n)
    X_stream   = transform_docs_to_csr(docs, analyzer_func, stream_vocab, stream_idf)

    # Vocabulary content comparison
    ref_set    = set(ref_vocab)
    stream_set = set(stream_vocab)
    missing    = ref_set - stream_set
    extra      = stream_set - ref_set
    idx_mismatch = [(t, ref_vocab[t], stream_vocab[t])
                    for t in ref_set & stream_set
                    if ref_vocab[t] != stream_vocab[t]]

    log(f"  reference vocab size     : {len(ref_vocab)}")
    log(f"  streaming vocab size     : {len(stream_vocab)}")
    log(f"  missing terms            : {len(missing)}")
    log(f"  extra terms              : {len(extra)}")
    log(f"  index mapping mismatches : {len(idx_mismatch)}")

    if idx_mismatch:
        log(f"  [FAIL] examples: {idx_mismatch[:5]}")
        log("  ABORT: index mismatches remain after alphabetical sort fix")
        _fout.close()
        sys.exit(1)
    else:
        log("  [PASS] index mapping mismatches = 0")

    # IDF comparison (no alignment needed now — indices identical)
    idf_diff      = np.abs(ref_idf - stream_idf)
    max_idf_diff  = float(np.max(idf_diff))
    mean_idf_diff = float(np.mean(idf_diff))
    n_idf_exceed  = int(np.sum(idf_diff > 1e-6))
    log(f"  IDF max diff             : {max_idf_diff:.4e}")
    log(f"  IDF mean diff            : {mean_idf_diff:.4e}")
    log(f"  IDF entries > 1e-6       : {n_idf_exceed}")

    # Matrix comparison (columns are now identically ordered)
    diff_data    = (X_ref - X_stream).data
    max_mat      = float(np.max(np.abs(diff_data))) if diff_data.size > 0 else 0.0
    mean_mat     = float(np.mean(np.abs(diff_data))) if diff_data.size > 0 else 0.0
    n_mat_exceed = int(np.sum(np.abs(diff_data) > 1e-6))
    log(f"  matrix max diff          : {max_mat:.4e}")
    log(f"  matrix mean diff         : {mean_mat:.4e}")
    log(f"  matrix entries > 1e-6    : {n_mat_exceed}")

    rn_ref    = np.sqrt(X_ref.multiply(X_ref).sum(axis=1)).A1
    rn_stream = np.sqrt(X_stream.multiply(X_stream).sum(axis=1)).A1
    log(f"  max row-norm diff        : {float(np.max(np.abs(rn_ref - rn_stream))):.4e}")
    log()


# =======================================================================
# SECTION B: BLOCKER 2 — True incremental CSR writer, 1M benchmark
# =======================================================================
log("=" * 70)
log("SECTION B — Blocker 2: Incremental CSR writer benchmark (1M rows)")
log("=" * 70)

N_1M = 1_000_000
log(f"Loading {N_1M} rows from S2+S3 train corpus ...")
rss_before_load = rss_mib()
df_1m = load_corpus(N_1M)
rss_after_load  = rss_mib()
log(f"  RSS before load  : {rss_before_load:.1f} MiB")
log(f"  RSS after load   : {rss_after_load:.1f} MiB")
log(f"  Corpus hash      : {corpus_hash(df_1m['entity_id'].to_list())}")
log()

CHUNK_SIZES = [25_000, 50_000, 100_000, 200_000]
OUT_BASE    = "incremental_csr_output"

for cfg in CHANNELS:
    log(f"{'=' * 60}")
    log(f"Channel: {cfg['name']}")
    log(f"{'=' * 60}")

    docs_1m = df_1m[cfg["col"]].to_list()
    eids_1m = df_1m["entity_id"].to_list()
    n_total = len(docs_1m)

    # ---- SECTION D: DF/vocabulary memory profiling (first pass) ----
    log(f"  [D] DF counting and vocabulary construction")
    rss_pre_df = rss_mib()
    t_df = time.time()

    ref_vec_1m = TfidfVectorizer(
        analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32,
    )
    ref_vec_1m.fit(docs_1m)  # fit-only for analyzer (does NOT store matrix)
    analyzer_func = ref_vec_1m.build_analyzer()

    # Full DF pass over 1M docs (streaming in 50k sub-chunks for measurement)
    df_counts_1m = {}
    DF_SUBCHUNK = 50_000
    rss_df_peak = rss_pre_df
    for i in range(0, n_total, DF_SUBCHUNK):
        for doc in docs_1m[i : i + DF_SUBCHUNK]:
            for term in set(analyzer_func(doc)):
                df_counts_1m[term] = df_counts_1m.get(term, 0) + 1
        rss_now = rss_mib()
        if rss_now > rss_df_peak:
            rss_df_peak = rss_now

    vocab_1m, _ = build_vocab_streaming(
        docs_1m, analyzer_func, n_total, cfg["min_df"], cfg["max_df"]
    )
    idf_1m = build_idf(vocab_1m, df_counts_1m, n_total)
    rss_post_vocab = rss_mib()
    elapsed_df = time.time() - t_df

    log(f"  [D] RSS before DF pass   : {rss_pre_df:.1f} MiB")
    log(f"  [D] RSS peak during DF   : {rss_df_peak:.1f} MiB")
    log(f"  [D] RSS after vocab build: {rss_post_vocab:.1f} MiB")
    log(f"  [D] DF/vocab runtime (s) : {elapsed_df:.2f}")
    log(f"  [D] Raw DF dict size     : {len(df_counts_1m)} terms")
    log(f"  [D] Filtered vocab size  : {len(vocab_1m)}")
    del df_counts_1m   # release DF counts before transform pass
    log()

    # ---- SECTION B: Incremental CSR benchmark ----
    for chunk_size in CHUNK_SIZES:
        out_dir = os.path.join(OUT_BASE, cfg["name"], f"chunk_{chunk_size}")
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)

        log(f"  Chunk size: {chunk_size}")
        rss_peak     = rss_mib()
        t_start      = time.time()
        total_nnz    = 0
        total_disk   = 0
        chunk_num    = 0
        largest_disk = 0

        n_chunks = (n_total + chunk_size - 1) // chunk_size

        for i in range(0, n_total, chunk_size):
            chunk_docs = docs_1m[i : i + chunk_size]
            chunk_eids = eids_1m[i : i + chunk_size]
            n_chunk    = len(chunk_docs)

            # Build CSR for this chunk ONLY
            data_vals, col_idx, indptr = [], [], [0]
            for doc in chunk_docs:
                term_counts = {}
                for term in analyzer_func(doc):
                    if term in vocab_1m:
                        term_counts[term] = term_counts.get(term, 0) + 1
                if term_counts:
                    cols = [vocab_1m[t] for t in term_counts]
                    tfs  = np.array([term_counts[t] for t in term_counts], dtype=np.float32)
                    vals = tfs * idf_1m[cols]
                    norm = np.linalg.norm(vals)
                    if norm > 0.0:
                        vals = vals / norm
                    data_vals.extend(vals.tolist())
                    col_idx.extend(cols)
                indptr.append(len(data_vals))

            # Convert to numpy BEFORE writing
            arr_data    = np.array(data_vals,  dtype=np.float32)
            arr_indices = np.array(col_idx,    dtype=np.int32)
            arr_indptr  = np.array(indptr,     dtype=np.int32)
            arr_eids    = np.array(chunk_eids, dtype=object)
            chunk_nnz   = arr_data.size

            # Measure RSS after building chunk (before freeing lists)
            rss_now = rss_mib()
            if rss_now > rss_peak:
                rss_peak = rss_now

            # Write to disk immediately
            chunk_path = os.path.join(out_dir, f"{chunk_num:06d}")
            os.makedirs(chunk_path, exist_ok=True)
            np.save(chunk_path + "/data.npy",      arr_data)
            np.save(chunk_path + "/indices.npy",   arr_indices)
            np.save(chunk_path + "/indptr.npy",    arr_indptr)
            np.save(chunk_path + "/entity_ids.npy",arr_eids)
            meta = {
                "chunk_num":   chunk_num,
                "n_rows":      n_chunk,
                "nnz":         int(chunk_nnz),
                "global_row_start": i,
                "global_row_end":   i + n_chunk,
                "n_features":  len(vocab_1m),
            }
            with open(chunk_path + "/metadata.json", "w") as mf:
                json.dump(meta, mf)

            # Compute disk size for this chunk
            chunk_disk = sum(
                os.path.getsize(chunk_path + "/" + fn)
                for fn in ["data.npy","indices.npy","indptr.npy","entity_ids.npy"]
            )
            if chunk_disk > largest_disk:
                largest_disk = chunk_disk
            total_nnz  += chunk_nnz
            total_disk += chunk_disk

            # RELEASE chunk memory — no global accumulation
            del data_vals, col_idx, indptr, arr_data, arr_indices, arr_indptr, arr_eids
            chunk_num += 1

        elapsed = time.time() - t_start

        log(f"    input rows          : {n_total}")
        log(f"    n_chunks            : {chunk_num}")
        log(f"    peak process RSS    : {rss_peak:.1f} MiB")
        log(f"    runtime (s)         : {elapsed:.2f}")
        log(f"    total NNZ           : {total_nnz}")
        log(f"    total disk (MiB)    : {total_disk / (1024*1024):.2f}")
        log(f"    largest chunk disk  : {largest_disk / (1024*1024):.2f} MiB")
        log()

        # ---- SECTION C: Reconstruction verification ----
        # Reload all chunks and verify total row count, NNZ, entity ID ordering
        log(f"  [C] Reconstruction verification (chunk_size={chunk_size})")
        recon_eids = []
        recon_nnz  = 0
        recon_rows = 0
        all_chunks = sorted(os.listdir(out_dir))
        for cn in all_chunks:
            cp = os.path.join(out_dir, cn)
            if not os.path.isdir(cp):
                continue
            ch_eids = np.load(cp + "/entity_ids.npy", allow_pickle=True).tolist()
            ch_data = np.load(cp + "/data.npy")
            recon_eids.extend(ch_eids)
            recon_nnz  += ch_data.size
            with open(cp + "/metadata.json") as mf:
                m = json.load(mf)
            recon_rows += m["n_rows"]

        eid_ok  = recon_eids == eids_1m
        rows_ok = recon_rows == n_total
        nnz_ok  = recon_nnz  == total_nnz
        log(f"    total rows reconstructed : {recon_rows} (expected {n_total}) -> {'PASS' if rows_ok else 'FAIL'}")
        log(f"    total NNZ reconstructed  : {recon_nnz}  (expected {total_nnz}) -> {'PASS' if nnz_ok else 'FAIL'}")
        log(f"    entity ID order match    : {'PASS' if eid_ok else 'FAIL'}")
        log()

        # Numerical verification on a 5k reference subset
        N_CHECK = 5_000
        log(f"  [C] Numerical check on first {N_CHECK} rows vs sklearn reference")
        ref_small = TfidfVectorizer(
            analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32,
        )
        # Fit on the FULL 1M docs to get the same vocab, transform only first N_CHECK
        ref_small.fit(docs_1m)
        X_ref_small = ref_small.transform(docs_1m[:N_CHECK])

        # Load first chunk(s) to cover N_CHECK rows
        recon_data, recon_indices, recon_indptr = [], [], [0]
        rows_loaded = 0
        for cn in sorted(os.listdir(out_dir)):
            cp = os.path.join(out_dir, cn)
            if not os.path.isdir(cp):
                continue
            ch_data    = np.load(cp + "/data.npy")
            ch_indices = np.load(cp + "/indices.npy")
            ch_indptr  = np.load(cp + "/indptr.npy")
            n_ch_rows  = len(ch_indptr) - 1
            need       = min(N_CHECK - rows_loaded, n_ch_rows)
            # Take first 'need' rows from this chunk
            end_ptr    = int(ch_indptr[need])
            recon_data.extend(ch_data[:end_ptr].tolist())
            recon_indices.extend(ch_indices[:end_ptr].tolist())
            for j in range(need):
                recon_indptr.append(recon_indptr[-1] + int(ch_indptr[j+1]) - int(ch_indptr[j]))
            rows_loaded += need
            if rows_loaded >= N_CHECK:
                break

        X_recon = sp.csr_matrix(
            (np.array(recon_data, dtype=np.float32),
             np.array(recon_indices, dtype=np.int32),
             np.array(recon_indptr,  dtype=np.int32)),
            shape=(N_CHECK, len(vocab_1m)),
        )
        diff = (X_ref_small - X_recon).data
        max_diff  = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
        n_exceed  = int(np.sum(np.abs(diff) > 1e-6))
        log(f"    max matrix diff vs sklearn : {max_diff:.4e}")
        log(f"    entries > 1e-6             : {n_exceed}")
        log(f"    numerical check            : {'PASS' if n_exceed == 0 else 'FAIL'}")
        log()

log("=" * 70)
log("ALL SECTIONS COMPLETE")
log(f"Full output saved to: {OUTFILE}")
log("=" * 70)
_fout.close()
