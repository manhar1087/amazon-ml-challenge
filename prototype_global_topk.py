# Prototype: Global Top‑K correctness verification
# ---------------------------------------------------
# This script implements the minimal deterministic test described in the
# user request.  It **does not** touch any production code – it only reads a
# tiny subset of the existing source parquet files, runs the reference
# monolithic retrieval, runs a chunked version with an explicit global‑
# top‑K merge, and prints a concise equality report.
# ---------------------------------------------------

import time
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# ---------------------------------------------------------------------
# 1️⃣ Load deterministic mini‑datasets
# ---------------------------------------------------------------------
# Source files (already present in the repository)
SOURCE1 = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet/source1.parquet"
SOURCE2 = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet/source2.parquet"
SOURCE3 = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet/source3.parquet"

# Load full tables (Polars is efficient for columnar I/O)
print("[prototype] Loading source parquet files …")
source1 = pl.read_parquet(SOURCE1)
source2 = pl.read_parquet(SOURCE2)
source3 = pl.read_parquet(SOURCE3)

# Deterministic subsets (first N rows – ordering is stable on disk)
N_QUERIES = 200
N_CORPUS   = 10_000
queries = source1.head(N_QUERIES).select(["entity_id", "latin_name"])
# Concatenate source2 and source3 then take the first N_CORPUS rows
corpus = pl.concat([source2, source3]).head(N_CORPUS).select(["entity_id", "latin_name"])

# Convert to Python lists for the TF‑IDF vectoriser
query_texts = queries["latin_name"].to_list()
corpus_texts = corpus["latin_name"].to_list()

# Keep the original global positions (0‑based) – this is the tie‑break key
global_positions = np.arange(len(corpus_texts), dtype=np.int32)
corpus_entity_ids = corpus["entity_id"].to_numpy()
query_entity_ids  = queries["entity_id"].to_numpy()

# ---------------------------------------------------------------------
# 2️⃣ Frozen TF‑IDF configuration (Task 03 word analyser)
# ---------------------------------------------------------------------
ANALYZER = "word"
NGRAM_RANGE = (1, 1)
MIN_DF = 2
MAX_DF = 0.01  # fraction – we keep the default behaviour of sklearn
K = 20  # small K for the prototype (still matches the reference default)

vectorizer = TfidfVectorizer(
    analyzer=ANALYZER,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_df=MAX_DF,
    dtype=np.float32,
)

# Fit on the *full* mini‑corpus (reference behaviour)
print("[prototype] Fitting TF‑IDF on the 10k mini‑corpus …")
X_corpus = vectorizer.fit_transform(corpus_texts)
X_queries = vectorizer.transform(query_texts)

# ---------------------------------------------------------------------
# 3️⃣ Reference monolithic retrieval (single call to sp_matmul_topn)
# ---------------------------------------------------------------------
print("[prototype] Running reference monolithic retrieval …")
start = time.time()
ref_sim = sp_matmul_topn(X_queries, X_corpus.T, K, sort=True)
ref_elapsed = time.time() - start

# Extract reference results: for each query row we store (score, entity_id, global_pos)
ref_candidates = []  # list of list per query
for row in range(ref_sim.shape[0]):
    s = ref_sim.data[ref_sim.indptr[row] : ref_sim.indptr[row + 1]]
    idx = ref_sim.indices[ref_sim.indptr[row] : ref_sim.indptr[row + 1]]
    # idx are column positions inside X_corpus (i.e. 0‑based global positions)
    # Build a list of (score, entity_id, global_pos)
    row_cands = list(zip(s, corpus_entity_ids[idx], global_positions[idx]))
    # Already sorted descending by score (sp_matmul_topn with sort=True)
    ref_candidates.append(row_cands)

# ---------------------------------------------------------------------
# 4️⃣ Chunked retrieval with explicit global‑top‑K merge
# ---------------------------------------------------------------------
print("[prototype] Running chunked retrieval with global‑top‑K merge …")
CHUNK_COUNT = 3
chunk_size = int(np.ceil(N_CORPUS / CHUNK_COUNT))

# Prepare per‑query min‑heaps (list of (score, entity_id, global_pos))
# We'll keep the list sorted ascending (smallest first) for easy replacement.
heaps = [[(float('-inf'), None, None) for _ in range(K)] for _ in range(N_QUERIES)]
# Actually we initialise as empty lists and will maintain size <= K
heaps = [[] for _ in range(N_QUERIES)]

chunk_start_time = time.time()
for chunk_idx in range(CHUNK_COUNT):
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, N_CORPUS)
    X_chunk = X_corpus[start_idx:end_idx]
    # Local top‑K retrieval for this chunk (no internal sorting – we will sort later)
    local_sim = sp_matmul_topn(X_queries, X_chunk.T, K, sort=False)
    # Walk the CSR rows and push into the per‑query heap
    for row in range(local_sim.shape[0]):
        data = local_sim.data[local_sim.indptr[row] : local_sim.indptr[row + 1]]
        idx = local_sim.indices[local_sim.indptr[row] : local_sim.indptr[row + 1]]
        for score, local_col in zip(data, idx):
            global_col = start_idx + int(local_col)
            entity_id = corpus_entity_ids[global_col]
            pos = global_positions[global_col]
            heap = heaps[row]
            if len(heap) < K:
                heap.append((score, entity_id, pos))
                if len(heap) == K:
                    heap.sort(key=lambda x: x[0])  # smallest first
            else:
                # heap[0] is the smallest (worst) score currently kept
                if score > heap[0][0] or (score == heap[0][0] and pos < heap[0][2]):
                    heap[0] = (score, entity_id, pos)
                    heap.sort(key=lambda x: x[0])
# After all chunks, each heap contains at most K best entries per query
# Sort each heap descending by score, then by global position for deterministic ties
chunk_elapsed = time.time() - chunk_start_time

chunked_candidates = []
for row in range(N_QUERIES):
    heap = heaps[row]
    heap.sort(key=lambda x: (-x[0], x[2]))  # descending score, ascending position
    chunked_candidates.append(heap)

# ---------------------------------------------------------------------
# 5️⃣ Equality verification
# ---------------------------------------------------------------------
print("[prototype] Verifying equality …")
matched_queries = 0
mismatched_queries = 0
mismatched_rows = 0
for q in range(N_QUERIES):
    ref = ref_candidates[q]
    chk = chunked_candidates[q]
    # Both lists should have length K (or less if corpus is tiny)
    if len(ref) != len(chk):
        mismatched_queries += 1
        continue
    # Compare element‑wise
    equal = True
    for (s_ref, id_ref, pos_ref), (s_chk, id_chk, pos_chk) in zip(ref, chk):
        if id_ref != id_chk or abs(s_ref - s_chk) > 1e-7 or pos_ref != pos_chk:
            equal = False
            mismatched_rows += 1
            break
    if equal:
        matched_queries += 1
    else:
        mismatched_queries += 1

print("--- Equality Report ---")
print(f"Total queries examined               : {N_QUERIES}")
print(f"Queries with exact match (all K)      : {matched_queries}")
print(f"Queries with any mismatch            : {mismatched_queries}")
print(f"Total mismatched candidate rows      : {mismatched_rows}")
print(f"Reference retrieval time (s)         : {ref_elapsed:.3f}")
print(f"Chunked retrieval + merge time (s)    : {chunk_elapsed:.3f}")

# ---------------------------------------------------------------------
# 6️⃣ Additional explicit tests
# ---------------------------------------------------------------------
print("[prototype] Running explicit multi‑chunk distribution test …")
# For this test we ensure that the best candidate for each query lives in a
# *different* chunk.  We pick the first K rows of the corpus (one per chunk)
# and copy them into each chunk, making them identical.
# To keep the code short we reuse the previously built TF‑IDF matrices.

# Create a very small artificial corpus: 3 rows, each duplicated across chunks
# We set K=2 for clarity.
K_test = 2
small_corpus_texts = ["identical text", "identical text", "identical text"]
vectorizer_test = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1, dtype=np.float32)
X_small = vectorizer_test.fit_transform(small_corpus_texts)
X_q_test = vectorizer_test.transform(["identical text"] * 5)  # 5 queries

# Chunk the small corpus into three single‑row chunks
chunks_test = [X_small[i:i+1] for i in range(3)]
heaps_test = [[] for _ in range(5)]
for ci, X_chunk in enumerate(chunks_test):
    sim = sp_matmul_topn(X_q_test, X_chunk.T, K_test, sort=False)
    for row in range(sim.shape[0]):
        data = sim.data[sim.indptr[row]:sim.indptr[row+1]]
        idx = sim.indices[sim.indptr[row]:sim.indptr[row+1]]
        for s, local_idx in zip(data, idx):
            global_idx = ci  # each chunk has a unique global row index
            heap = heaps_test[row]
            if len(heap) < K_test:
                heap.append((s, global_idx))
                if len(heap) == K_test:
                    heap.sort(key=lambda x: x[0])
            else:
                if s > heap[0][0]:
                    heap[0] = (s, global_idx)
                    heap.sort(key=lambda x: x[0])
# After merge, each query should have *exactly* K=2 distinct candidates
# drawn from the three chunks (the two highest‑scoring rows).
print("[prototype] Multi‑chunk distribution test results (per query top‑2 global row indices):")
for row, heap in enumerate(heaps_test):
    heap.sort(key=lambda x: -x[0])
    print(f"  Query {row}: {heap}")

print("[prototype] Explicit tie test (duplicate rows) …")
# Duplicate the first row of the corpus to create an exact tie
X_dup = X_corpus.copy()
X_dup = sp.vstack([X_dup, X_dup[0]])  # add a duplicate as last row
entity_ids_dup = np.append(corpus_entity_ids, corpus_entity_ids[0])
positions_dup = np.append(global_positions, N_CORPUS)  # new position
# Run a single‑chunk retrieval on this augmented corpus and verify that the
# duplicate rows appear next to each other after sorting by score then pos.
sim_dup = sp_matmul_topn(X_queries, X_dup.T, K, sort=True)
# Extract the first query's top K rows
row = 0
s = sim_dup.data[sim_dup.indptr[row]:sim_dup.indptr[row+1]]
idx = sim_dup.indices[sim_dup.indptr[row]:sim_dup.indptr[row+1]]
print("  First query top‑K after duplicate insertion (entity_id, position):")
for score, i in zip(s, idx):
    print(f"    score={score:.6f}, entity_id={entity_ids_dup[i]}, pos={positions_dup[i]}")

print("[prototype] Done.")
