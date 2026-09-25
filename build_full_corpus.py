import os, sys, json, time, shutil, heapq, gc
import numpy as np
import pyarrow.parquet as pq
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import polars as pl

PROC = psutil.Process(os.getpid())
def get_rss(): return PROC.memory_info().rss / (1024 * 1024)

BASE = "work/parquet"
SOURCE2 = BASE + "/train/source2.parquet"
SOURCE3 = BASE + "/train/source3.parquet"
FILES = [SOURCE2, SOURCE3]
TOTAL_EXPECTED_ROWS = 10_320_219

CHANNELS = [
    {"name":"name_word",    "analyzer":"word",    "ngram_range":(1,1), "min_df":2, "max_df":0.01,  "col":"latin_name",    "K":50},
    {"name":"address_word", "analyzer":"word",    "ngram_range":(1,2), "min_df":2, "max_df":0.02,  "col":"latin_address", "K":20},
    {"name":"name_char",    "analyzer":"char_wb", "ngram_range":(3,4), "min_df":2, "max_df":0.005, "col":"latin_name",    "K":20},
]

SETUP_BASE = "retrieval_setup"
CSR_BASE = "incremental_csr_output"

OUTFILE = "build_full_corpus_output.txt"
_f = open(OUTFILE, "w", encoding="utf-8")
def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg, flush=True)
    _f.write(msg + "\n")
    _f.flush()

def iter_corpus_stream(files, col_name, batch_size=100_000, limit=None, yield_eids=False):
    rows_yielded = 0
    cols = [col_name, "entity_id"] if yield_eids else [col_name]
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            col_data = batch[col_name].to_pylist()
            if yield_eids:
                eid_data = batch["entity_id"].to_pylist()
            
            if limit is not None:
                rem = limit - rows_yielded
                if len(col_data) > rem:
                    col_data = col_data[:rem]
                    if yield_eids: eid_data = eid_data[:rem]
            
            if yield_eids:
                yield col_data, eid_data
            else:
                yield col_data
                
            rows_yielded += len(col_data)
            if limit is not None and rows_yielded >= limit:
                return

def write_batch(local_df, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for term in sorted(local_df.keys()):
            f.write(f"{json.dumps(term)}\t{local_df[term]}\n")

def read_batch(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            t_str, c_str = line.rsplit('\t', 1)
            yield (json.loads(t_str), int(c_str))

def build_vocab_idf(files, col_name, n_samples, min_df, max_df, analyzer_func, channel_name):
    log(f"  Building vocab & IDF for {channel_name}...")
    tmp_dir = os.path.join(SETUP_BASE, "tmp_df_batches")
    if os.path.exists(tmp_dir): shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    batch_files = []
    batch_idx = 0
    t0 = time.time()
    rss_peak = get_rss()
    
    # 1. Batching
    for docs_batch in iter_corpus_stream(files, col_name, batch_size=100_000):
        local_df = {}
        for doc in docs_batch:
            if doc is None: continue
            for term in set(analyzer_func(doc)):
                local_df[term] = local_df.get(term, 0) + 1
        
        filepath = os.path.join(tmp_dir, f"batch_{batch_idx:04d}.tsv")
        write_batch(local_df, filepath)
        batch_files.append(filepath)
        local_df.clear()
        batch_idx += 1
        cur_rss = get_rss()
        if cur_rss > rss_peak: rss_peak = cur_rss
        
    log(f"  DF batched in {time.time()-t0:.1f}s. Merging {len(batch_files)} files...")
    
    # 2. Merging
    generators = [read_batch(f) for f in batch_files]
    merged_stream = heapq.merge(*generators)
    
    min_df_abs = min_df if isinstance(min_df, int) else int(np.ceil(min_df * n_samples))
    max_df_abs = max_df if isinstance(max_df, int) else int(np.floor(max_df * n_samples))
    
    filtered_terms = []
    current_term = None
    current_count = 0
    
    for term, count in merged_stream:
        if term == current_term:
            current_count += count
        else:
            if current_term is not None:
                if min_df_abs <= current_count <= max_df_abs:
                    filtered_terms.append((current_term, current_count))
            current_term = term
            current_count = count
            
    if current_term is not None:
        if min_df_abs <= current_count <= max_df_abs:
            filtered_terms.append((current_term, current_count))
            
    # Guarantee exact alphabetical indexing (fixes Blocker 1)
    filtered_terms.sort(key=lambda x: x[0])
    vocab = {t: i for i, (t, c) in enumerate(filtered_terms)}
    df_counts = np.array([c for t, c in filtered_terms], dtype=np.float64)
    
    idf = (np.log((1.0 + n_samples) / (1.0 + df_counts)) + 1.0).astype(np.float32)
    
    out_dir = os.path.join(SETUP_BASE, channel_name)
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    np.save(os.path.join(out_dir, "idf.npy"), idf)
    
    shutil.rmtree(tmp_dir)
    cur_rss = get_rss()
    if cur_rss > rss_peak: rss_peak = cur_rss
    log(f"  Vocab ({len(vocab)} terms) built in {time.time()-t0:.1f}s. Peak RSS={rss_peak:.1f} MiB")
    return vocab, idf, rss_peak, len(vocab)

def build_csr_chunks(files, col_name, vocab, idf, analyzer_func, chunk_size, channel_name):
    log(f"  Building CSR chunks for {channel_name}...")
    out_dir = os.path.join(CSR_BASE, channel_name, f"chunk_{chunk_size}")
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    t0 = time.time()
    rss_peak = get_rss()
    
    current_chunk = 0
    global_row_start = 0
    
    dv, ci, ip, eids = [], [], [0], []
    total_nnz = 0
    
    for docs_batch, eids_batch in iter_corpus_stream(files, col_name, batch_size=20000, yield_eids=True):
        for i in range(len(docs_batch)):
            doc = docs_batch[i]
            eid = eids_batch[i]
            
            tc = {}
            if doc is not None:
                for t in analyzer_func(doc):
                    if t in vocab:
                        tc[t] = tc.get(t, 0) + 1
            
            if tc:
                cols = [vocab[t] for t in tc]
                vals = np.array([tc[t] for t in tc], dtype=np.float32) * idf[cols]
                norm = np.linalg.norm(vals)
                if norm > 0: vals /= norm
                dv.extend(vals.tolist())
                ci.extend(cols)
                
            ip.append(len(dv))
            eids.append(eid)
            
            # Flush chunk
            if len(eids) == chunk_size:
                c_dir = os.path.join(out_dir, f"chunk_{current_chunk:04d}")
                os.makedirs(c_dir)
                
                np.save(c_dir + "/data.npy", np.array(dv, dtype=np.float32))
                np.save(c_dir + "/indices.npy", np.array(ci, dtype=np.int32))
                np.save(c_dir + "/indptr.npy", np.array(ip, dtype=np.int32))
                np.save(c_dir + "/entity_ids.npy", np.array(eids, dtype=str))
                
                meta = {
                    "global_row_start": global_row_start,
                    "n_rows": len(eids),
                    "nnz": len(dv)
                }
                with open(c_dir + "/metadata.json", "w") as f:
                    json.dump(meta, f)
                    
                total_nnz += len(dv)
                global_row_start += len(eids)
                current_chunk += 1
                
                dv, ci, ip, eids = [], [], [0], []
                gc.collect()
                cur_rss = get_rss()
                if cur_rss > rss_peak: rss_peak = cur_rss
                
    # Final chunk
    if len(eids) > 0:
        c_dir = os.path.join(out_dir, f"chunk_{current_chunk:04d}")
        os.makedirs(c_dir)
        
        np.save(c_dir + "/data.npy", np.array(dv, dtype=np.float32))
        np.save(c_dir + "/indices.npy", np.array(ci, dtype=np.int32))
        np.save(c_dir + "/indptr.npy", np.array(ip, dtype=np.int32))
        np.save(c_dir + "/entity_ids.npy", np.array(eids, dtype=str))
        
        meta = {
            "global_row_start": global_row_start,
            "n_rows": len(eids),
            "nnz": len(dv)
        }
        with open(c_dir + "/metadata.json", "w") as f:
            json.dump(meta, f)
            
        total_nnz += len(dv)
        global_row_start += len(eids)
        current_chunk += 1
        cur_rss = get_rss()
        if cur_rss > rss_peak: rss_peak = cur_rss

    disk_size = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, filenames in os.walk(out_dir)
        for f in filenames
    )
    
    log(f"  Chunks built: {current_chunk}. Total NNZ: {total_nnz}. Time: {time.time()-t0:.1f}s. Peak RSS: {rss_peak:.1f} MiB")
    return current_chunk, total_nnz, rss_peak, disk_size

def get_dir_size(start_path):
    total = 0
    for dirpath, _, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

def run_build():
    log("="*70)
    log("FULL 10.32M CORPUS BUILD - TASK 03")
    log("="*70)
    
    t_start_global = time.time()
    
    cnt2 = pl.scan_parquet(SOURCE2).select(pl.len()).collect().item()
    cnt3 = pl.scan_parquet(SOURCE3).select(pl.len()).collect().item()
    total_rows = cnt2 + cnt3
    log(f"Source2 rows: {cnt2}")
    log(f"Source3 rows: {cnt3}")
    log(f"Total Corpus Rows: {total_rows}")
    
    if total_rows != TOTAL_EXPECTED_ROWS:
        log(f"ERROR: Expected {TOTAL_EXPECTED_ROWS} rows, got {total_rows}")
        sys.exit(1)
        
    manifest = {
        "status": "IN_PROGRESS",
        "source2_path": SOURCE2,
        "source3_path": SOURCE3,
        "source2_rows": cnt2,
        "source3_rows": cnt3,
        "total_corpus_rows": total_rows,
        "chunk_size": 50000,
        "channels": {}
    }
    
    for cfg in CHANNELS:
        ch_name = cfg["name"]
        log(f"\n--- Channel: {ch_name} ---")
        
        vec_dummy = TfidfVectorizer(analyzer=cfg["analyzer"], ngram_range=cfg["ngram_range"])
        analyzer_func = vec_dummy.build_analyzer()
        
        t0 = time.time()
        vocab, idf, df_rss, v_size = build_vocab_idf(
            FILES, cfg["col"], total_rows, cfg["min_df"], cfg["max_df"], analyzer_func, ch_name
        )
        t_df = time.time() - t0
        
        t0 = time.time()
        n_chunks, nnz, csr_rss, csr_disk = build_csr_chunks(
            FILES, cfg["col"], vocab, idf, analyzer_func, 50000, ch_name
        )
        t_csr = time.time() - t0
        
        manifest["channels"][ch_name] = {
            "config": cfg,
            "vocab_size": v_size,
            "df_runtime_s": t_df,
            "df_peak_rss_mib": df_rss,
            "csr_runtime_s": t_csr,
            "csr_peak_rss_mib": csr_rss,
            "num_chunks": n_chunks,
            "total_nnz": nnz,
            "csr_disk_usage_bytes": csr_disk
        }
        
    manifest["status"] = "SUCCESS"
    manifest["total_runtime_s"] = time.time() - t_start_global
    
    with open("full_build_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    log("\nBuild successful. Manifest saved to full_build_manifest.json")
    log(f"Total time: {manifest['total_runtime_s']:.1f}s")

if __name__ == '__main__':
    run_build()
