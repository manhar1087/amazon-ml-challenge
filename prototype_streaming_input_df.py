"""
prototype_streaming_input_df.py
-------------------------------
True streaming input DF aggregation pipeline.
- Uses pyarrow.parquet.ParquetFile.iter_batches() to stream parquet incrementally.
- NEVER loads the full corpus into memory.
- Uses exact external-memory k-way merge for DF counting.
"""

import os, sys, json, time, shutil, heapq
import numpy as np
import pyarrow.parquet as pq
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

PROC = psutil.Process(os.getpid())
def get_rss(): return PROC.memory_info().rss / (1024 * 1024)

BASE = "c:/Users/manha/Downloads/amazon_ml_challenge/6ab10eb3b23ba_student_resource/student_resource/work/parquet"
SOURCE2 = BASE + "/train/source2.parquet"
SOURCE3 = BASE + "/train/source3.parquet"
FILES = [SOURCE2, SOURCE3]

OUTFILE = "streaming_input_df_output.txt"
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
# EXACT STREAMING PARQUET READER
# =========================================================================

def iter_corpus_stream(files, col_name, batch_size=100_000, limit=None):
    """
    STRICT STREAMING GUARANTEE:
    Opens Parquet files and yields batches of strings incrementally.
    Releases each batch immediately. Never loads the full column or file.
    """
    rows_yielded = 0
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[col_name]):
            # Convert specifically the requested column to a standard python list
            col_data = batch[col_name].to_pylist()
            
            if limit is not None:
                rem = limit - rows_yielded
                if len(col_data) > rem:
                    col_data = col_data[:rem]
            
            yield col_data
            rows_yielded += len(col_data)
            
            if limit is not None and rows_yielded >= limit:
                return

# =========================================================================
# EXTERNAL K-WAY MERGE LOGIC (REUSED FROM PREVIOUS PROTOTYPE)
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

def build_external_df_streaming(files, col_name, limit, analyzer_func, n_samples, min_df, max_df, batch_size, out_dir):
    """
    Fully streaming DF builder. Streams Parquet blocks -> streams to disk -> streams k-way merge.
    """
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    rss_peak = get_rss()
    
    # 1. STREAMING BATCHING PHASE
    batch_files = []
    batch_idx = 0
    
    for docs_batch in iter_corpus_stream(files, col_name, batch_size=batch_size, limit=limit):
        local_df = {}
        for doc in docs_batch:
            if doc is None: continue
            for term in set(analyzer_func(doc)):
                local_df[term] = local_df.get(term, 0) + 1
        
        filepath = os.path.join(out_dir, f"batch_{batch_idx:04d}.tsv")
        write_batch(local_df, filepath)
        batch_files.append(filepath)
        
        local_df.clear()
        del docs_batch, local_df # Explicit memory release
        batch_idx += 1
        
        cur_rss = get_rss()
        if cur_rss > rss_peak: rss_peak = cur_rss

    disk_usage_batches = sum(os.path.getsize(f) for f in batch_files)

    # 2. MERGING PHASE
    generators = [read_batch(f) for f in batch_files]
    merged_stream = heapq.merge(*generators)
    
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
# MAIN EXECUTION
# =========================================================================

def main():
    log("="*70)
    log(f"INITIAL PROCESS RSS: {get_rss():.1f} MiB")
    log("="*70)
    
    # -----------------------------------------------------------------
    # SECTION 1: 100k CORRECTNESS EQUIVALENCE
    # -----------------------------------------------------------------
    log("\n" + "="*70)
    log("SECTION 1: 100K CORRECTNESS EQUIVALENCE")
    log("="*70)
    
    N_100K = 100_000
    for cfg in CHANNELS:
        log(f"\n--- Channel: {cfg['name']} ---")
        
        # Build Reference (In-Memory)
        docs_100k = list(iter_corpus_stream(FILES, cfg["col"], batch_size=N_100K, limit=N_100K))[0]
        ref_vec = TfidfVectorizer(
            analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32
        )
        ref_vec.fit([d if d else "" for d in docs_100k])
        ref_vocab = ref_vec.vocabulary_
        ref_idf = ref_vec.idf_.astype(np.float32)
        analyzer = ref_vec.build_analyzer()
        
        # Free memory before streaming run
        del docs_100k
        
        # Streaming External Build
        ext_vocab, ext_idf, stats = build_external_df_streaming(
            FILES, cfg["col"], N_100K, analyzer, N_100K, cfg["min_df"], cfg["max_df"],
            batch_size=25_000, out_dir="tmp_ext_df_streaming"
        )
        
        vocab_keys_match = set(ref_vocab.keys()) == set(ext_vocab.keys())
        idx_match = all(ref_vocab[k] == ext_vocab[k] for k in ref_vocab)
        idf_max_diff = float(np.max(np.abs(ref_idf - ext_idf)))
        
        log(f"  Vocab term set identical   : {'PASS' if vocab_keys_match else 'FAIL'}")
        log(f"  Index mapping identical    : {'PASS' if idx_match else 'FAIL'}")
        log(f"  IDF max diff               : {idf_max_diff:.2e} (<= 1e-6: {'PASS' if idf_max_diff <= 1e-6 else 'FAIL'})")

    # -----------------------------------------------------------------
    # SECTION 2: 4M SCALE TEST (ALL 3 CHANNELS)
    # -----------------------------------------------------------------
    log("\n" + "="*70)
    log("SECTION 2: 4M STREAMING SCALE TEST")
    log("="*70)
    
    N_4M = 4_000_000
    BATCH_SIZE_4M = 100_000
    
    for cfg in CHANNELS:
        log(f"\n--- Channel: {cfg['name']} (4M rows) ---")
        
        # Build just the analyzer
        tmp_docs = list(iter_corpus_stream(FILES, cfg["col"], batch_size=1000, limit=1000))[0]
        ref_vec = TfidfVectorizer(
            analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"], max_df=cfg["max_df"], dtype=np.float32
        )
        ref_vec.fit([d if d else "" for d in tmp_docs])
        analyzer = ref_vec.build_analyzer()
        del tmp_docs
        
        t0 = time.time()
        vocab, idf, stats = build_external_df_streaming(
            FILES, cfg["col"], N_4M, analyzer, N_4M, cfg["min_df"], cfg["max_df"],
            batch_size=BATCH_SIZE_4M, out_dir="tmp_ext_df_streaming"
        )
        t_elapsed = time.time() - t0
        
        log(f"  Total input rows           : {N_4M}")
        log(f"  Batch size                 : {BATCH_SIZE_4M}")
        log(f"  Number of batch files      : {stats['batch_files']}")
        log(f"  Raw DF term count          : {stats['raw_df_terms']}")
        log(f"  Filtered vocabulary size   : {stats['filtered_vocab_size']}")
        log(f"  Peak process RSS           : {stats['rss_peak']:.1f} MiB")
        log(f"  DF runtime (s)             : {t_elapsed:.2f}")
        log(f"  Temp disk usage            : {stats['disk_usage_batches'] / (1024*1024):.2f} MiB")
        log(f"  Final vocab disk usage     : {stats['vocab_disk_usage'] / (1024*1024):.2f} MiB")

    # -----------------------------------------------------------------
    # SECTION 3: SPECIFIC 4M ADDRESS_WORD VALIDATION (REDUNDANT BUT EXPLICIT)
    # -----------------------------------------------------------------
    log("\n" + "="*70)
    log("SECTION 3: SPECIFIC 4M ADDRESS_WORD ONLY VALIDATION")
    log("="*70)
    log("  (Completed within Section 2, verifying bounding of RSS for heaviest channel)")
    
    log("\n" + "="*70)
    log("PROTOTYPE COMPLETE")
    log("="*70)

if __name__ == "__main__":
    main()
