# Deterministic Streaming TF‑IDF Prototype
# ------------------------------------------------------------
# This script provides a reproducible run on a fixed 200 k‑row
# S2+S3 mini‑corpus, reports detailed differences between a sklearn
# reference implementation and the streaming implementation, and
# measures peak RAM for various chunk sizes.
# ------------------------------------------------------------

import os
import time
import hashlib
import json
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import psutil

# ------------------------------------------------------------------
# 1️⃣  Paths – use the exact source2 and source3 parquet files
# ------------------------------------------------------------------
SOURCE2 = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/source2.parquet"
SOURCE3 = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/source3.parquet"

# ------------------------------------------------------------------
# 2️⃣  Frozen TF‑IDF channel configuration (must match Task 03)
# ------------------------------------------------------------------
CHANNELS = [
    {"name": "name_word", "analyzer": "word", "ngram_range": (1, 1), "min_df": 2, "max_df": 0.01},
    {"name": "address_word", "analyzer": "word", "ngram_range": (1, 2), "min_df": 2, "max_df": 0.02},
    {"name": "name_char", "analyzer": "char_wb", "ngram_range": (3, 4), "min_df": 2, "max_df": 0.005},
]

# ------------------------------------------------------------------
# 3️⃣  Corpus selection – deterministic 200 k rows from S2+S3
# ------------------------------------------------------------------
N_CORPUS = 200_000  # exact number of rows
CHUNK_ROWS_DEFAULT = 20_000

print("[prototype] Loading explicit source2 and source3 parquet files …")
# Lazy scans – only the columns we need
lf2 = pl.scan_parquet(SOURCE2).select(["entity_id", "latin_name"])
lf3 = pl.scan_parquet(SOURCE3).select(["entity_id", "latin_name"])
# Row counts for the manifest
cnt2 = lf2.select(pl.count()).collect().item()
cnt3 = lf3.select(pl.count()).collect().item()
print(f"[manifest] source2 rows={cnt2}, source3 rows={cnt3}")
# Concatenate in a fixed order (source2 then source3)
full_lf = pl.concat([lf2, lf3])
# Pull the first N_CORPUS rows – this defines the deterministic mini‑corpus
mini_lf = full_lf.head(N_CORPUS).collect()
corpus_texts = mini_lf["latin_name"].to_list()
entity_ids = mini_lf["entity_id"].to_numpy()
print(f"[prototype] Mini‑corpus loaded: {len(corpus_texts)} documents")
# Compute a SHA‑256 hash of the ordered entity IDs for reproducibility verification
hash_obj = hashlib.sha256()
for eid in entity_ids:
    hash_obj.update(str(eid).encode("utf-8"))
corpus_hash = hash_obj.hexdigest()
print(f"[manifest] corpus hash (entity_id order): {corpus_hash}")

# ------------------------------------------------------------------
# Helper to stream tokenised documents in chunks
# ------------------------------------------------------------------
def stream_documents(analyzer, docs, chunk_size):
    for i in range(0, len(docs), chunk_size):
        chunk = docs[i : i + chunk_size]
        tokenised = [analyzer(doc) for doc in chunk]
        yield tokenised, i, i + len(chunk)

# ------------------------------------------------------------------
# 4️⃣  Detailed per‑channel comparison
# ------------------------------------------------------------------
for cfg in CHANNELS:
    print("\n[prototype] ==== Channel: {} ====".format(cfg['name']))
    start_total = time.time()
    # ---- Reference sklearn ----
    ref_vec = TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        dtype=np.float32,
    )
    X_ref = ref_vec.fit_transform(corpus_texts)
    ref_vocab = ref_vec.vocabulary_
    ref_idf = ref_vec.idf_.astype(np.float32)
    print(f"[reference] vocab size: {len(ref_vocab)}")
    # ---- Streaming implementation ----
    analyzer_func = ref_vec.build_analyzer()
    n_samples = len(corpus_texts)
    # First pass – document frequencies
    df_counts = {}
    for tokenised_chunk, _, _ in stream_documents(analyzer_func, corpus_texts, CHUNK_ROWS_DEFAULT):
        for tokens in tokenised_chunk:
            for term in set(tokens):
                df_counts[term] = df_counts.get(term, 0) + 1
    # Apply min_df / max_df thresholds exactly as sklearn does
    if isinstance(cfg["min_df"], float):
        min_df_abs = int(np.ceil(cfg["min_df"] * n_samples))
    else:
        min_df_abs = cfg["min_df"]
    if isinstance(cfg["max_df"], float):
        max_df_abs = int(np.floor(cfg["max_df"] * n_samples))
    else:
        max_df_abs = cfg["max_df"]
    vocab = {}
    for term, df in df_counts.items():
        if df >= min_df_abs and df <= max_df_abs:
            vocab[term] = len(vocab)
    print(f"[streaming] vocab size after thresholds: {len(vocab)}")
    # IDF computation – same formula as sklearn
    idf = np.log((1 + n_samples) / (1 + np.array([df_counts[t] for t in sorted(vocab, key=vocab.get)]))) + 1.0
    idf = idf.astype(np.float32)
    # Second pass – build CSR matrix
    data_vals = []
    indices = []
    indptr = [0]
    for tokenised_chunk, _, _ in stream_documents(analyzer_func, corpus_texts, CHUNK_ROWS_DEFAULT):
        for tokens in tokenised_chunk:
            term_counts = {}
            for term in tokens:
                if term in vocab:
                    term_counts[term] = term_counts.get(term, 0) + 1
            row_data = []
            row_indices = []
            for term, count in term_counts.items():
                col_idx = vocab[term]
                row_indices.append(col_idx)
                row_data.append(count)
            if row_data:
                row_data = np.array(row_data, dtype=np.float32)
                row_data = row_data * idf[[vocab[t] for t in term_counts.keys()]]
                norm = np.linalg.norm(row_data)
                if norm != 0.0:
                    row_data = row_data / norm
                data_vals.extend(row_data.tolist())
                indices.extend(row_indices)
            indptr.append(len(data_vals))
    X_stream = sparse.csr_matrix(
        (np.array(data_vals, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)),
        shape=(n_samples, len(vocab)),
    )
    print(f"[streaming] matrix shape: {X_stream.shape}, nnz: {X_stream.nnz}")
    # ---- Vocabulary contents comparison ----
    ref_set = set(ref_vocab.keys())
    stream_set = set(vocab.keys())
    missing = ref_set - stream_set
    extra = stream_set - ref_set
    print(f"[compare] vocab missing (ref‑not‑stream): {len(missing)}")
    print(f"[compare] vocab extra (stream‑not‑ref): {len(extra)}")
    # Index mapping mismatches (show up to 5 examples)
    mismatches = []
    for term in ref_set & stream_set:
        if ref_vocab[term] != vocab[term]:
            mismatches.append((term, ref_vocab[term], vocab[term]))
    print(f"[compare] index mapping mismatches: {len(mismatches)}")
    if mismatches:
        print("  examples:", mismatches[:5])
    # ---- IDF comparison ----
    ref_to_stream = np.full(len(ref_vocab), -1, dtype=np.int32)
    for term, ref_idx in ref_vocab.items():
        ref_to_stream[ref_idx] = vocab[term]
    idf_stream_aligned = idf[ref_to_stream]
    idf_diff_vec = np.abs(ref_idf - idf_stream_aligned)
    max_idf_diff = np.max(idf_diff_vec)
    mean_idf_diff = np.mean(idf_diff_vec)
    n_idf_exceed = np.sum(idf_diff_vec > 1e-6)
    print(f"[compare] IDF max diff: {max_idf_diff:.2e}, mean diff: {mean_idf_diff:.2e}, >1e-6 count: {n_idf_exceed}")
    # ---- Matrix comparison ----
    X_stream_aligned = X_stream[:, ref_to_stream]
    diff_data = (X_ref - X_stream_aligned).data
    max_abs_diff = np.max(np.abs(diff_data)) if diff_data.size > 0 else 0.0
    mean_abs_diff = np.mean(np.abs(diff_data)) if diff_data.size > 0 else 0.0
    n_diff_entries = np.sum(np.abs(diff_data) > 1e-6)
    print(f"[compare] matrix max entry diff: {max_abs_diff:.2e}, mean diff: {mean_abs_diff:.2e}, >1e-6 count: {n_diff_entries}")
    # Row‑norm verification
    row_norms_ref = np.sqrt(X_ref.multiply(X_ref).sum(axis=1)).A1
    row_norms_stream = np.sqrt(X_stream_aligned.multiply(X_stream_aligned).sum(axis=1)).A1
    norm_diff = np.max(np.abs(row_norms_ref - row_norms_stream))
    print(f"[compare] max row‑norm diff: {norm_diff:.2e}")
    # ---- Runtime & RAM ----
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    elapsed = time.time() - start_total
    print(f"[stats] runtime (s): {elapsed:.3f}, peak RAM (MiB): {mem_mb:.1f}\n")

# ------------------------------------------------------------------
# 5️⃣  Memory benchmark for different chunk sizes (using name_word channel)
# ------------------------------------------------------------------
print("\n[benchmark] Memory usage for various chunk sizes (name_word channel)\n")
benchmark_cfg = CHANNELS[0]  # name_word
for chunk in [25_000, 50_000, 100_000, 200_000]:
    print(f"[benchmark] Chunk size: {chunk}")
    start = time.time()
    ref_vec = TfidfVectorizer(
        analyzer=benchmark_cfg["analyzer"],
        ngram_range=benchmark_cfg["ngram_range"],
        min_df=benchmark_cfg["min_df"],
        max_df=benchmark_cfg["max_df"],
        dtype=np.float32,
    )
    analyzer_func = ref_vec.build_analyzer()
    n_samples = len(corpus_texts)
    # First pass DF
    df_counts = {}
    for tokenised_chunk, _, _ in stream_documents(analyzer_func, corpus_texts, chunk):
        for tokens in tokenised_chunk:
            for term in set(tokens):
                df_counts[term] = df_counts.get(term, 0) + 1
    # Thresholds
    min_df_abs = benchmark_cfg["min_df"] if isinstance(benchmark_cfg["min_df"], int) else int(np.ceil(benchmark_cfg["min_df"] * n_samples))
    max_df_abs = benchmark_cfg["max_df"] if isinstance(benchmark_cfg["max_df"], int) else int(np.floor(benchmark_cfg["max_df"] * n_samples))
    vocab = {term: i for i, (term, df) in enumerate(df_counts.items()) if min_df_abs <= df <= max_df_abs}
    # IDF
    idf = np.log((1 + n_samples) / (1 + np.array([df_counts[t] for t in sorted(vocab, key=vocab.get)]))) + 1.0
    idf = idf.astype(np.float32)
    # Second pass – build CSR (no need to keep the matrix after)
    data_vals = []
    indices = []
    indptr = [0]
    for tokenised_chunk, _, _ in stream_documents(analyzer_func, corpus_texts, chunk):
        for tokens in tokenised_chunk:
            term_counts = {}
            for term in tokens:
                if term in vocab:
                    term_counts[term] = term_counts.get(term, 0) + 1
            row_data = []
            row_indices = []
            for term, count in term_counts.items():
                col_idx = vocab[term]
                row_indices.append(col_idx)
                row_data.append(count)
            if row_data:
                row_data = np.array(row_data, dtype=np.float32)
                row_data = row_data * idf[[vocab[t] for t in term_counts.keys()]]
                norm = np.linalg.norm(row_data)
                if norm != 0.0:
                    row_data = row_data / norm
                data_vals.extend(row_data.tolist())
                indices.extend(row_indices)
            indptr.append(len(data_vals))
    del data_vals, indices, indptr, vocab, df_counts, idf
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    elapsed = time.time() - start
    print(f"    runtime (s): {elapsed:.3f}, peak RAM (MiB): {mem_mb:.1f}\n")

# ------------------------------------------------------------------
# 6️⃣  Manifest file for reproducibility
# ------------------------------------------------------------------
manifest = {
    "source_files": [SOURCE2, SOURCE3],
    "rows_per_file": {os.path.basename(SOURCE2): cnt2, os.path.basename(SOURCE3): cnt3},
    "total_rows_selected": N_CORPUS,
    "corpus_hash": corpus_hash,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
manifest_path = "deterministic_corpus_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"[manifest] written to {manifest_path}\n")
# End of script
