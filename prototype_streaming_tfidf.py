# Prototype: Streaming TF‑IDF construction vs sklearn reference
# ---------------------------------------------------------------
# Goal: Verify that a streaming vocabulary/IDF build followed by chunked TF‑IDF
# transformation produces exactly the same matrix (up to a tiny tolerance)
# as sklearn's TfidfVectorizer on the same corpus subset.
# ---------------------------------------------------------------

import time
import os
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import psutil

# ---------------------------------------------------------------------
# 1️⃣  Paths to parquet sources (S2 and S3) – they reside under work/parquet
# ---------------------------------------------------------------------
SOURCE2 = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/source2.parquet"
SOURCE3 = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/source3.parquet"

# ---------------------------------------------------------------------
# 2️⃣  Settings for the three TF‑IDF channels (frozen configuration)
# ---------------------------------------------------------------------
CHANNELS = [
    {
        "name": "name_word",
        "analyzer": "word",
        "ngram_range": (1, 1),
        "min_df": 2,
        "max_df": 0.01,
    },
    {
        "name": "address_word",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.02,
    },
    {
        "name": "name_char",
        "analyzer": "char_wb",
        "ngram_range": (3, 4),
        "min_df": 2,
        "max_df": 0.005,
    },
]

# ---------------------------------------------------------------------
# 3️⃣  Load a deterministic 200 k‑row mini‑corpus (concatenated S2+S3)
# ---------------------------------------------------------------------
N_CORPUS = 200_000  # rows to use for the prototype
CHUNK_ROWS = 20_000  # size of each processing chunk

print("[prototype] Loading S2+S3 and extracting mini-corpus ...")
# Load lazily, then collect only the needed columns (entity_id, latin_name)
# Load all parquet files under work/parquet recursively
lf_all = pl.scan_parquet("c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet/**/*.parquet").select(["entity_id", "latin_name"])
full_lf = lf_all
mini_lf = full_lf.head(N_CORPUS).collect()
corpus_texts = mini_lf["latin_name"].to_list()
entity_ids = mini_lf["entity_id"].to_numpy()
print(f"[prototype] Mini-corpus loaded: {len(corpus_texts)} documents")

# ---------------------------------------------------------------------
# 4️⃣  Helper: stream documents through a given analyzer and yield token lists
# ---------------------------------------------------------------------
def stream_documents(analyzer, docs, chunk_size):
    """Yield tokenised documents in chunks.
    `analyzer` is a callable returning a list of tokens for a raw string.
    """
    for i in range(0, len(docs), chunk_size):
        chunk = docs[i : i + chunk_size]
        tokenised = [analyzer(doc) for doc in chunk]
        yield tokenised, i, i + len(chunk)

# ---------------------------------------------------------------------
# 5️⃣  Process each channel
# ---------------------------------------------------------------------
for cfg in CHANNELS:
    print("\n[prototype] ==== Channel: {} ====".format(cfg["name"]))
    start_total = time.time()

    # ---------------------------------------------------------------
    # 5.1 Build a sklearn TfidfVectorizer for reference (full fit)
    # ---------------------------------------------------------------
    ref_vec = TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        dtype=np.float32,
    )
    X_ref = ref_vec.fit_transform(corpus_texts)  # dense CSR matrix
    ref_vocab = ref_vec.vocabulary_
    ref_idf = ref_vec.idf_
    print("[reference] vocab size:", len(ref_vocab))
    print("[reference] matrix shape:", X_ref.shape, "nnz:", X_ref.nnz)

    # ---------------------------------------------------------------
    # 5.2 Streaming implementation
    # ---------------------------------------------------------------
    # Obtain the exact sklearn analyzer function (ensures tokenisation matches)
    analyzer_func = ref_vec.build_analyzer()
    n_samples = len(corpus_texts)
    # First pass: compute document frequencies (DF)
    df_counts = {}
    for tokenised_chunk, start_idx, end_idx in stream_documents(analyzer_func, corpus_texts, CHUNK_ROWS):
        for tokens in tokenised_chunk:
            unique_terms = set(tokens)
            for term in unique_terms:
                df_counts[term] = df_counts.get(term, 0) + 1
    # Apply min_df / max_df thresholds exactly as sklearn does
    min_df = cfg["min_df"]
    max_df = cfg["max_df"]
    if isinstance(min_df, float):
        min_df_abs = int(np.ceil(min_df * n_samples))
    else:
        min_df_abs = min_df
    if isinstance(max_df, float):
        max_df_abs = int(np.floor(max_df * n_samples))
    else:
        max_df_abs = max_df
    # Build final vocabulary (term -> index) respecting order of appearance
    vocab = {}
    for term, df in df_counts.items():
        if df >= min_df_abs and df <= max_df_abs:
            vocab[term] = len(vocab)  # assign incremental index
    print("[streaming] vocab size after thresholds:", len(vocab))
    # Compute IDF using sklearn's formula
    idf = np.log((1 + n_samples) / (1 + np.array([df_counts[t] for t in sorted(vocab, key=vocab.get)]))) + 1.0
    idf = idf.astype(np.float32)
    # Second pass: transform documents into CSR data arrays
    data_vals = []
    indices = []
    indptr = [0]
    for tokenised_chunk, start_idx, end_idx in stream_documents(analyzer_func, corpus_texts, CHUNK_ROWS):
        for tokens in tokenised_chunk:
            term_counts = {}
            for term in tokens:
                if term in vocab:
                    term_counts[term] = term_counts.get(term, 0) + 1
            # Compute TF (raw count) -> multiply by IDF later
            row_data = []
            row_indices = []
            for term, count in term_counts.items():
                col_idx = vocab[term]
                row_indices.append(col_idx)
                row_data.append(count)
            # Convert to numpy for speed
            if len(row_data) > 0:
                row_data = np.array(row_data, dtype=np.float32)
                # Apply IDF weighting
                row_data = row_data * idf[[vocab[t] for t in term_counts.keys()]]
                # L2‑normalize the row (sklearn default norm='l2')
                norm = np.linalg.norm(row_data)
                if norm != 0.0:
                    row_data = row_data / norm
                data_vals.extend(row_data.tolist())
                indices.extend(row_indices)
            indptr.append(len(data_vals))
    # Assemble CSR matrix
    X_stream = sparse.csr_matrix((np.array(data_vals, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)), shape=(n_samples, len(vocab)))
    print("[streaming] matrix shape:", X_stream.shape, "nnz:", X_stream.nnz)

    # ---------------------------------------------------------------
    # 5.3 Compare reference and streaming results
    # ---------------------------------------------------------------
    # Vocabulary comparison (size and content)
    vocab_match = (len(ref_vocab) == len(vocab)) and all(term in vocab for term in ref_vocab)
    print("[compare] vocab size match:", len(ref_vocab) == len(vocab))
    # Align columns if ordering differs
    # Build mapping from ref index -> streaming index
    ref_to_stream = np.full(len(ref_vocab), -1, dtype=np.int32)
    for term, ref_idx in ref_vocab.items():
        ref_to_stream[ref_idx] = vocab[term]
    # Reorder streaming matrix columns to reference order for direct comparison
    X_stream_aligned = X_stream[:, ref_to_stream]
    # Compare IDF vectors (using same ordering as reference)
    idf_ref = ref_vec.idf_.astype(np.float32)
    idf_stream = idf[ref_to_stream]
    idf_diff = np.max(np.abs(idf_ref - idf_stream))
    print("[compare] max IDF diff:", idf_diff)
    # Compare sparse matrix values (tolerance 1e-6)
    diff = (X_ref - X_stream_aligned).data
    max_abs_diff = np.max(np.abs(diff)) if diff.size > 0 else 0.0
    print("[compare] max matrix entry diff:", max_abs_diff)
    # Row‑norm verification (should all be ~1.0)
    row_norms_ref = np.sqrt(X_ref.multiply(X_ref).sum(axis=1)).A1
    row_norms_stream = np.sqrt(X_stream_aligned.multiply(X_stream_aligned).sum(axis=1)).A1
    norm_diff = np.max(np.abs(row_norms_ref - row_norms_stream))
    print("[compare] max row-norm diff:", norm_diff)

    # ---------------------------------------------------------------
    # 5️⃣  Memory / runtime statistics
    # ---------------------------------------------------------------
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    elapsed = time.time() - start_total
    print("[stats] runtime (s):", round(elapsed, 3), "peak RAM (MiB):", round(mem_mb, 1))
    # Disk usage for CSR representation
    # Save arrays temporarily to measure size (will be deleted afterwards)
    np.save("tmp_data.npy", data_vals)
    np.save("tmp_indices.npy", indices)
    np.save("tmp_indptr.npy", indptr)
    total_bytes = os.path.getsize("tmp_data.npy") + os.path.getsize("tmp_indices.npy") + os.path.getsize("tmp_indptr.npy")
    print("[stats] CSR disk usage (MiB):", round(total_bytes / (1024 * 1024), 2))
    # Clean up temporary files
    os.remove("tmp_data.npy")
    os.remove("tmp_indices.npy")
    os.remove("tmp_indptr.npy")

    print("[prototype] ==== Channel {} completed ====".format(cfg["name"]))

print("\nAll channels processed. If any diff > tolerance, review implementation.")
