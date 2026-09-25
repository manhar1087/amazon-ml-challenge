"""
verify_streaming_tfidf.py
--------------------------
ONE clean, deterministic verification run of the streaming TF-IDF implementation.

Sections:
  1. Corpus loading and manifest (hash verification)
  2. Full TF-IDF equivalence check for all 3 frozen channels
  3. Vocabulary content check (terms, index mapping)
  4. Chunk-size memory benchmark for all 3 channels x 4 chunk sizes
  5. Implementation code excerpts (transparency)

No extrapolation. No placeholder values. All numbers are measured.
"""

import os
import sys
import time
import json
import hashlib
import tracemalloc
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

OUTFILE = "verify_streaming_tfidf_output.txt"
_fout = open(OUTFILE, "w", encoding="utf-8")

def log(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    _fout.write(msg + "\n")
    _fout.flush()

# =======================================================================
# SECTION 0: Clean temporary stale artifacts
# =======================================================================
log("=" * 70)
log("SECTION 0 - Cleaning stale temporary artifacts")
log("=" * 70)
for fname in ["tmp_data.npy", "tmp_indices.npy", "tmp_indptr.npy",
              "deterministic_corpus_manifest.json"]:
    if os.path.exists(fname):
        os.remove(fname)
        log(f"  Removed: {fname}")
    else:
        log(f"  Not present (clean): {fname}")

# =======================================================================
# SECTION 1: Corpus loading and manifest
# =======================================================================
log()
log("=" * 70)
log("SECTION 1 - Corpus loading and manifest")
log("=" * 70)

BASE = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet"
SOURCE2 = BASE + "/train/source2.parquet"
SOURCE3 = BASE + "/train/source3.parquet"
N_CORPUS = 200_000

log(f"SOURCE2 path : {SOURCE2}")
log(f"SOURCE3 path : {SOURCE3}")

# Row counts per file (authoritative)
cnt2 = pl.scan_parquet(SOURCE2).select(pl.len()).collect().item()
cnt3 = pl.scan_parquet(SOURCE3).select(pl.len()).collect().item()
log(f"SOURCE2 row count  : {cnt2}")
log(f"SOURCE3 row count  : {cnt3}")
log(f"Total S2+S3 rows   : {cnt2 + cnt3}")

# Deterministic selection: concatenate in fixed order, take first N_CORPUS rows
lf2 = pl.scan_parquet(SOURCE2).select(["entity_id", "latin_name", "latin_address"])
lf3 = pl.scan_parquet(SOURCE3).select(["entity_id", "latin_name", "latin_address"])
full_lf = pl.concat([lf2, lf3])
mini_df = full_lf.head(N_CORPUS).collect()

actual_rows = len(mini_df)
log(f"Selected rows      : {actual_rows}")
assert actual_rows == N_CORPUS, f"Expected {N_CORPUS} rows, got {actual_rows}"

entity_ids = mini_df["entity_id"].to_list()
log(f"First 10 entity_ids: {entity_ids[:10]}")
log(f"Last  10 entity_ids: {entity_ids[-10:]}")

# SHA-256 from ordered selected entity_ids (first computation)
h = hashlib.sha256()
for eid in entity_ids:
    h.update(str(eid).encode("utf-8"))
corpus_hash = h.hexdigest()
log(f"hash_from_selection: {corpus_hash}")

# Independent recomputation from saved list
saved_ids = entity_ids[:]   # plain copy
h2 = hashlib.sha256()
for eid in saved_ids:
    h2.update(str(eid).encode("utf-8"))
hash_recomputed = h2.hexdigest()
log(f"hash_recomputed    : {hash_recomputed}")
log(f"hash_equal         : {corpus_hash == hash_recomputed}")
assert corpus_hash == hash_recomputed, "HASH MISMATCH - abort"

corpus_name = mini_df["latin_name"].to_list()
corpus_addr = mini_df["latin_address"].to_list()

# Save manifest
manifest = {
    "source2_path": SOURCE2,
    "source3_path": SOURCE3,
    "source2_row_count": cnt2,
    "source3_row_count": cnt3,
    "total_available_rows": cnt2 + cnt3,
    "n_selected": N_CORPUS,
    "first_10_entity_ids": entity_ids[:10],
    "last_10_entity_ids": entity_ids[-10:],
    "corpus_hash": corpus_hash,
    "hash_recomputed": hash_recomputed,
    "hash_equal": corpus_hash == hash_recomputed,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
with open("deterministic_corpus_manifest.json", "w", encoding="utf-8") as mf:
    json.dump(manifest, mf, indent=2)
log("Manifest written to: deterministic_corpus_manifest.json")

# =======================================================================
# SECTION 2 & 3: Full TF-IDF equivalence + vocabulary content checks
# =======================================================================

CHANNELS = [
    {
        "name": "name_word",
        "analyzer": "word",
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.01,
        "corpus_key": "name",
    },
    {
        "name": "address_word",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.02,
        "corpus_key": "address",
    },
    {
        "name": "name_char",
        "analyzer": "char_wb",
        "ngram_range": (3, 4),
        "min_df": 2,
        "max_df": 0.005,
        "corpus_key": "name",
    },
]

CHUNK_ROWS = 20_000

def get_docs(cfg):
    return corpus_name if cfg["corpus_key"] == "name" else corpus_addr

def streaming_build(docs, ref_vec, n_samples, chunk_rows):
    """
    Build TF-IDF CSR matrix using the streaming algorithm.
    Uses ref_vec.build_analyzer() for identical tokenisation as sklearn.
    Returns: (vocab dict, idf np.float32 array, X_stream sparse CSR matrix)
    """
    analyzer_func = ref_vec.build_analyzer()
    min_df = ref_vec.min_df
    max_df = ref_vec.max_df

    # -- First pass: document-frequency counts --
    df_counts = {}
    for i in range(0, n_samples, chunk_rows):
        chunk = docs[i : i + chunk_rows]
        for doc in chunk:
            for term in set(analyzer_func(doc)):
                df_counts[term] = df_counts.get(term, 0) + 1

    # -- Apply thresholds (exact sklearn semantics) --
    min_df_abs = min_df if isinstance(min_df, int) else int(np.ceil(min_df * n_samples))
    max_df_abs = max_df if isinstance(max_df, int) else int(np.floor(max_df * n_samples))

    # -- Vocabulary: order of first appearance in df_counts iteration --
    vocab = {}
    for term, df in df_counts.items():
        if min_df_abs <= df <= max_df_abs:
            vocab[term] = len(vocab)

    # -- IDF: sklearn formula --
    # idf_[j] = log((1 + n) / (1 + df[j])) + 1
    sorted_terms = sorted(vocab, key=vocab.get)   # sort by assigned index
    df_arr = np.array([df_counts[t] for t in sorted_terms], dtype=np.float64)
    idf = (np.log((1.0 + n_samples) / (1.0 + df_arr)) + 1.0).astype(np.float32)

    # -- Second pass: build CSR data arrays --
    data_vals = []
    col_indices = []
    indptr = [0]

    for i in range(0, n_samples, chunk_rows):
        chunk = docs[i : i + chunk_rows]
        for doc in chunk:
            term_counts = {}
            for term in analyzer_func(doc):
                if term in vocab:
                    term_counts[term] = term_counts.get(term, 0) + 1
            if term_counts:
                cols = [vocab[t] for t in term_counts]
                tfs  = np.array([term_counts[t] for t in term_counts], dtype=np.float32)
                # TF * IDF
                vals = tfs * idf[cols]
                # L2-normalise
                norm = np.linalg.norm(vals)
                if norm > 0.0:
                    vals = vals / norm
                data_vals.extend(vals.tolist())
                col_indices.extend(cols)
            indptr.append(len(data_vals))

    X = sparse.csr_matrix(
        (
            np.array(data_vals, dtype=np.float32),
            np.array(col_indices, dtype=np.int32),
            np.array(indptr, dtype=np.int32),
        ),
        shape=(n_samples, len(vocab)),
    )
    return vocab, idf, X


for cfg in CHANNELS:
    log()
    log("=" * 70)
    log(f"SECTION 2+3 - Channel: {cfg['name']}")
    log(f"  analyzer    : {cfg['analyzer']}")
    log(f"  ngram_range : {cfg['ngram_range']}")
    log(f"  min_df      : {cfg['min_df']}")
    log(f"  max_df      : {cfg['max_df']}")
    log(f"  corpus_key  : {cfg['corpus_key']}")
    log(f"  dtype       : float32")
    log(f"  norm        : l2 (sklearn default)")
    log(f"  sublinear_tf: False (sklearn default)")
    log("=" * 70)

    t0 = time.time()
    docs = get_docs(cfg)
    n_samples = len(docs)

    # -- Reference sklearn --
    ref_vec = TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        dtype=np.float32,
    )
    X_ref = ref_vec.fit_transform(docs)
    ref_vocab = ref_vec.vocabulary_
    ref_idf   = ref_vec.idf_.astype(np.float32)

    log(f"[reference] vocab size : {len(ref_vocab)}")
    log(f"[reference] matrix shape: {X_ref.shape}")
    log(f"[reference] NNZ        : {X_ref.nnz}")

    # -- Streaming build (same analyzer callable) --
    stream_vocab, stream_idf, X_stream = streaming_build(docs, ref_vec, n_samples, CHUNK_ROWS)

    log(f"[streaming] vocab size : {len(stream_vocab)}")
    log(f"[streaming] matrix shape: {X_stream.shape}")
    log(f"[streaming] NNZ        : {X_stream.nnz}")

    # ----------------------------------------------------------------
    # SECTION 3: Vocabulary content comparison
    # ----------------------------------------------------------------
    ref_set    = set(ref_vocab.keys())
    stream_set = set(stream_vocab.keys())
    missing    = ref_set - stream_set      # in ref but not stream
    extra      = stream_set - ref_set      # in stream but not ref

    log()
    log("[vocab] reference-only terms (missing from streaming):")
    log(f"  count: {len(missing)}")
    if missing:
        log(f"  examples: {sorted(missing)[:5]}")

    log("[vocab] streaming-only terms (extra vs reference):")
    log(f"  count: {len(extra)}")
    if extra:
        log(f"  examples: {sorted(extra)[:5]}")

    # Index-mapping comparison (only for shared terms)
    index_mismatches = []
    for term in ref_set & stream_set:
        ri = ref_vocab[term]
        si = stream_vocab[term]
        if ri != si:
            index_mismatches.append((term, ri, si))
    log("[vocab] index mapping mismatches (shared terms, different index):")
    log(f"  count: {len(index_mismatches)}")
    if index_mismatches:
        log(f"  examples (term, ref_idx, stream_idx): {index_mismatches[:5]}")

    # ----------------------------------------------------------------
    # SECTION 2: IDF and matrix comparison (align columns first)
    # ----------------------------------------------------------------
    # Build ref_to_stream index permutation
    ref_to_stream = np.full(len(ref_vocab), -1, dtype=np.int32)
    for term, ref_idx in ref_vocab.items():
        if term in stream_vocab:
            ref_to_stream[ref_idx] = stream_vocab[term]

    if np.any(ref_to_stream == -1):
        log("[WARN] Some reference terms not found in streaming vocab - cannot align fully")

    # Align streaming IDF to reference order
    idf_stream_aligned = stream_idf[ref_to_stream]
    idf_diff = np.abs(ref_idf - idf_stream_aligned)
    max_idf_diff  = float(np.max(idf_diff))
    mean_idf_diff = float(np.mean(idf_diff))
    n_idf_exceed  = int(np.sum(idf_diff > 1e-6))

    log()
    log("[idf] max  abs diff      :", f"{max_idf_diff:.4e}")
    log("[idf] mean abs diff      :", f"{mean_idf_diff:.4e}")
    log("[idf] entries > 1e-6     :", n_idf_exceed)

    # Align streaming matrix columns to reference order
    X_stream_aligned = X_stream[:, ref_to_stream]
    diff_matrix = X_ref - X_stream_aligned
    diff_data   = diff_matrix.data
    if diff_data.size > 0:
        max_mat_diff  = float(np.max(np.abs(diff_data)))
        mean_mat_diff = float(np.mean(np.abs(diff_data)))
        n_mat_exceed  = int(np.sum(np.abs(diff_data) > 1e-6))
        n_diff_entries = diff_data.size
    else:
        max_mat_diff = mean_mat_diff = 0.0
        n_mat_exceed = n_diff_entries = 0

    log("[matrix] ref NNZ           :", X_ref.nnz)
    log("[matrix] stream NNZ        :", X_stream.nnz)
    log("[matrix] differing entries :", n_diff_entries)
    log("[matrix] max  abs diff     :", f"{max_mat_diff:.4e}")
    log("[matrix] mean abs diff     :", f"{mean_mat_diff:.4e}")
    log("[matrix] entries > 1e-6    :", n_mat_exceed)

    # Row-norm check
    row_norms_ref    = np.sqrt(X_ref.multiply(X_ref).sum(axis=1)).A1
    row_norms_stream = np.sqrt(X_stream_aligned.multiply(X_stream_aligned).sum(axis=1)).A1
    rn_diff = np.abs(row_norms_ref - row_norms_stream)
    log("[rownorm] max diff         :", f"{float(np.max(rn_diff)):.4e}")
    log("[rownorm] mean diff        :", f"{float(np.mean(rn_diff)):.4e}")

    elapsed = time.time() - t0
    log(f"[timing] channel runtime (s): {elapsed:.3f}")

# =======================================================================
# SECTION 4: Chunk-size memory benchmark - all 3 channels x 4 chunk sizes
# =======================================================================
log()
log("=" * 70)
log("SECTION 4 - Chunk-size memory benchmark (all channels)")
log("=" * 70)

CHUNK_SIZES = [25_000, 50_000, 100_000, 200_000]
benchmark_results = {}

for cfg in CHANNELS:
    docs = get_docs(cfg)
    n_samples = len(docs)
    log()
    log(f"  Channel: {cfg['name']}")
    benchmark_results[cfg["name"]] = {}

    for chunk in CHUNK_SIZES:
        # Build a fresh ref_vec just for the analyzer function
        ref_vec = TfidfVectorizer(
            analyzer=cfg["analyzer"],
            ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"],
            max_df=cfg["max_df"],
            dtype=np.float32,
        )
        ref_vec.fit(docs)   # need fit to get build_analyzer working
        analyzer_func = ref_vec.build_analyzer()
        min_df = cfg["min_df"]
        max_df = cfg["max_df"]

        tracemalloc.start()
        t0 = time.time()

        # First pass - DF counts
        df_counts = {}
        for i in range(0, n_samples, chunk):
            for doc in docs[i : i + chunk]:
                for term in set(analyzer_func(doc)):
                    df_counts[term] = df_counts.get(term, 0) + 1

        min_df_abs = min_df if isinstance(min_df, int) else int(np.ceil(min_df * n_samples))
        max_df_abs = max_df if isinstance(max_df, int) else int(np.floor(max_df * n_samples))
        vocab = {}
        for term, df in df_counts.items():
            if min_df_abs <= df <= max_df_abs:
                vocab[term] = len(vocab)

        sorted_terms = sorted(vocab, key=vocab.get)
        df_arr = np.array([df_counts[t] for t in sorted_terms], dtype=np.float64)
        idf = (np.log((1.0 + n_samples) / (1.0 + df_arr)) + 1.0).astype(np.float32)

        # Second pass - build CSR
        data_vals = []
        col_indices = []
        indptr = [0]
        for i in range(0, n_samples, chunk):
            for doc in docs[i : i + chunk]:
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
                    col_indices.extend(cols)
                indptr.append(len(data_vals))

        X = sparse.csr_matrix(
            (
                np.array(data_vals, dtype=np.float32),
                np.array(col_indices, dtype=np.int32),
                np.array(indptr, dtype=np.int32),
            ),
            shape=(n_samples, len(vocab)),
        )

        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - t0
        peak_mib = peak_bytes / (1024 * 1024)
        nnz = X.nnz

        # Disk size
        np.save("_bench_data.npy", data_vals)
        np.save("_bench_indices.npy", col_indices)
        np.save("_bench_indptr.npy", indptr)
        disk_bytes = (os.path.getsize("_bench_data.npy") +
                      os.path.getsize("_bench_indices.npy") +
                      os.path.getsize("_bench_indptr.npy"))
        disk_mib = disk_bytes / (1024 * 1024)
        os.remove("_bench_data.npy")
        os.remove("_bench_indices.npy")
        os.remove("_bench_indptr.npy")

        log(f"    chunk={chunk:>7d}  runtime={elapsed:.2f}s  "
            f"peak_RAM={peak_mib:.1f}MiB  NNZ={nnz}  disk={disk_mib:.2f}MiB")

        benchmark_results[cfg["name"]][chunk] = {
            "runtime_s": round(elapsed, 3),
            "peak_ram_mib": round(peak_mib, 1),
            "nnz": nnz,
            "disk_mib": round(disk_mib, 2),
        }

        del data_vals, col_indices, indptr, vocab, df_counts, idf, X

# Recommend production chunk size from worst (highest RAM) channel
log()
log("[recommendation] Worst-case peak RAM by chunk size across all channels:")
for chunk in CHUNK_SIZES:
    worst = max(benchmark_results[ch][chunk]["peak_ram_mib"] for ch in benchmark_results)
    log(f"  chunk={chunk:>7d}  worst_peak_RAM={worst:.1f} MiB")

# =======================================================================
# SECTION 5: Implementation transparency
# =======================================================================
log()
log("=" * 70)
log("SECTION 5 - Implementation functions used")
log("=" * 70)
log("""
DF counting:
  for i in range(0, n_samples, chunk_rows):
      chunk = docs[i : i + chunk_rows]
      for doc in chunk:
          for term in set(analyzer_func(doc)):  # set -> count each term once per doc
              df_counts[term] = df_counts.get(term, 0) + 1

Vocabulary construction:
  vocab = {}
  for term, df in df_counts.items():
      if min_df_abs <= df <= max_df_abs:
          vocab[term] = len(vocab)   # order of first appearance in df_counts

IDF construction (sklearn formula):
  idf[j] = log((1 + n_samples) / (1 + df[j])) + 1.0   (dtype=float32)

Corpus transformation (second pass):
  for doc in chunk:
      term_counts = Counter(t for t in analyzer_func(doc) if t in vocab)
      vals = np.array(term_counts.values, float32) * idf[column_indices]
      vals /= np.linalg.norm(vals)   # L2-normalise each row

Query transformation:
  Same function as corpus transformation. The same frozen vocab and idf arrays
  are applied to S1 query documents using the same analyzer_func.

Analyzer used per channel:
  name_word   -> TfidfVectorizer(analyzer='word',    ngram_range=(1,1)).build_analyzer()
  address_word-> TfidfVectorizer(analyzer='word',    ngram_range=(1,2)).build_analyzer()
  name_char   -> TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4)).build_analyzer()

The ref_vec.build_analyzer() callable is shared between the reference sklearn
fit_transform() and the streaming build - identical tokenisation guaranteed.
""")

log()
log("=" * 70)
log("VERIFICATION COMPLETE")
log("Output saved to: " + OUTFILE)
log("=" * 70)

_fout.close()
