import os, json, gc, time, shutil
import polars as pl
import numpy as np

def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# We import the existing blocks and TF-IDF tools
from src.data.loader import load_all_data
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from src.blocking.cross_script import cross_script_name_blocking
from src.blocking.abbreviation import abbreviation_blocking
from src.blocking.union import union_candidates
from rapidfuzz import fuzz
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

def transform_docs(docs, vocab, idf, analyzer_func):
    n = len(docs)
    n_feat = len(vocab)
    dv, ci, ip = [], [], [0]
    for doc in docs:
        tc = {}
        for t in analyzer_func(doc):
            if t in vocab: tc[t] = tc.get(t, 0) + 1
        if tc:
            cols = [vocab[t] for t in tc]
            vals = np.array([tc[t] for t in tc], dtype=np.float32) * idf[cols]
            norm = np.linalg.norm(vals)
            if norm > 0: vals /= norm
            dv.extend(vals.tolist())
            ci.extend(cols)
        ip.append(len(dv))
    return sp.csr_matrix((np.array(dv, dtype=np.float32),
                          np.array(ci, dtype=np.int32),
                          np.array(ip, dtype=np.int32)), shape=(n, n_feat))

def get_chunk_dirs(channel_name, chunk_size, chunk_dir="incremental_csr_output"):
    p = os.path.join(chunk_dir, channel_name, f"chunk_{chunk_size}")
    return sorted(
        os.path.join(p,d) for d in os.listdir(p)
        if os.path.isdir(os.path.join(p,d))
    )

def chunked_topk(Q, chunk_dirs, n_features, K, s1_ids, corpus_entity_ids, query_batch=500):
    n_q = Q.shape[0]
    best_scores  = np.full((n_q, K), -np.inf, dtype=np.float64)
    best_gidx    = np.full((n_q, K), -1,      dtype=np.int64)

    for chunk_path in chunk_dirs:
        with open(chunk_path+"/metadata.json") as f: meta = json.load(f)
        g_start   = meta["global_row_start"]
        n_chunk   = meta["n_rows"]
        
        C = sp.csr_matrix(
            (np.load(chunk_path+"/data.npy"),
             np.load(chunk_path+"/indices.npy"),
             np.load(chunk_path+"/indptr.npy")),
            shape=(n_chunk, n_features))

        for q0 in range(0, n_q, query_batch):
            q1    = min(q0+query_batch, n_q)
            Qb    = Q[q0:q1]
            sim   = (Qb @ C.T).toarray() 
            
            for qi in range(q1-q0):
                q_global = q0+qi
                sim_row  = sim[qi].astype(np.float64)
                
                cur_scores = best_scores[q_global]
                cur_idx = best_gidx[q_global]
                
                # Combine old + new
                all_s = np.concatenate([cur_scores, sim_row])
                all_i = np.concatenate([cur_idx, np.arange(g_start, g_start+n_chunk, dtype=np.int64)])
                
                # Keep top K
                # Ignore invalid (-inf or -1)
                valid = all_i >= 0
                all_s = all_s[valid]
                all_i = all_i[valid]
                
                if len(all_s) > 0:
                    keep_k = min(K, len(all_s))
                    # argpartition is faster than sort
                    top_k_pos = np.argpartition(all_s, -keep_k)[-keep_k:]
                    # sort the top K to have them strictly descending
                    top_k_sorted_pos = top_k_pos[np.argsort(-all_s[top_k_pos])]
                    
                    best_scores[q_global][:keep_k] = all_s[top_k_sorted_pos]
                    best_gidx[q_global][:keep_k]   = all_i[top_k_sorted_pos]

    out_s1 = []
    out_s23 = []
    for i in range(n_q):
        s1_e = s1_ids[i]
        for j in range(K):
            idx = best_gidx[i, j]
            score = best_scores[i, j]
            if idx >= 0 and score > 0:
                out_s1.append(s1_e)
                out_s23.append(corpus_entity_ids[idx])
                
    return out_s1, out_s23

def run_tfidf_chunked_local(s1, col, analyzer, ngram, k, channel_name, base_channel, corpus_eids, split_name, setup_dir="retrieval_setup"):
    cache_dir = "work/task06_candidate_optimization/cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split_name}_{channel_name}_{k}.parquet")
    
    sdir = os.path.join(setup_dir, base_channel)
    with open(os.path.join(sdir, "vocab.json")) as f:
        vocab_size = len(json.load(f))
        
    if os.path.exists(cache_path):
        print(f"[{channel_name}] Loading cached Top-{k} retrieval for {split_name}...")
        df = pl.read_parquet(cache_path)
        return df, vocab_size
        
    print(f"[{channel_name}] Loading precomputed vocab and IDF...")
    with open(os.path.join(sdir, "vocab.json")) as f:
        vocab = json.load(f)
    idf = np.load(os.path.join(sdir, "idf.npy"))
    
    vec_dummy = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram)
    analyzer_func = vec_dummy.build_analyzer()
    
    s1_texts = s1[col].fill_null("").to_list()
    print(f"[{channel_name}] Transforming S1 Queries...")
    X_q = transform_docs(s1_texts, vocab, idf, analyzer_func)
    
    print(f"[{channel_name}] Vocab size: {vocab_size}")
    
    chunk_dirs = get_chunk_dirs(base_channel, chunk_size=50000)
    print(f"[{channel_name}] Executing chunked Top-{k} retrieval over {len(chunk_dirs)} chunks...")
    
    out_s1, out_s23 = chunked_topk(X_q, chunk_dirs, vocab_size, k, s1["entity_id"].to_list(), corpus_eids)
    
    df = pl.DataFrame({
        "source1_entity_id": out_s1,
        "candidate_entity_id": out_s23,
        "retrieval_channels": [channel_name] * len(out_s1)
    })
    
    print(f"[{channel_name}] Saving cache for {split_name} Top-{k}...")
    df.write_parquet(cache_path)
    
    del X_q
    gc.collect()
    return df, vocab_size


def postal_house_partial_name_blocking(s1_lf: pl.LazyFrame, s23_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    New Block: Recovers candidates that have an exact postal code & house number match, 
    but only a partial name match (e.g. fuzzy token set ratio > 60).
    We do an exact join on postal_code and house_number to limit the cross-product, 
    then filter by string similarity.
    """
    s1 = s1_lf.select(["entity_id", "postal_code", "house_number", "latin_name"]).filter(
        pl.col("postal_code").is_not_null() & (pl.col("postal_code") != "") &
        pl.col("house_number").is_not_null() & (pl.col("house_number") != "") &
        pl.col("latin_name").is_not_null()
    )
    s23 = s23_lf.select(["entity_id", "postal_code", "house_number", "latin_name"]).filter(
        pl.col("postal_code").is_not_null() & (pl.col("postal_code") != "") &
        pl.col("house_number").is_not_null() & (pl.col("house_number") != "") &
        pl.col("latin_name").is_not_null()
    )
    
    joined = s1.join(s23, on=["postal_code", "house_number"], how="inner", suffix="_right")
    
    # Filter identical entities (if any leak in)
    joined = joined.filter(pl.col("entity_id") != pl.col("entity_id_right"))
    
    # Filter by fuzzy string match using rapidfuzz token_set_ratio
    # This is expensive, so we only run it on the joined subset
    joined = joined.collect()
    
    def fuzz_match(c1, c2):
        res = []
        for x, y in zip(c1, c2):
            res.append(fuzz.token_set_ratio(str(x), str(y)) > 60)
        return pl.Series(res, dtype=pl.Boolean)
        
    joined = joined.with_columns(
        fuzz_match(joined["latin_name"].to_list(), joined["latin_name_right"].to_list()).alias("fuzzy_match")
    ).filter(pl.col("fuzzy_match"))
    
    return joined.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id"),
        pl.lit("postal_house_fuzzy_name").alias("retrieval_channels")
    ])


def run():
    print("Loading Task 04 splits...")
    with open("work/task04/train_val_ids.json", "r") as f:
        splits = json.load(f)
    train_s1_ids = splits["train_ids"]
    val_s1_ids = sorted(splits["val_ids"])
    
    np.random.seed(42)
    tune_s1_ids = set(np.random.choice(val_s1_ids, size=len(val_s1_ids)//2, replace=False).tolist())
    eval_s1_ids = [x for x in val_s1_ids if x not in tune_s1_ids]
    tune_s1_ids = list(tune_s1_ids)
    
    print("Loading Ground Truth...")
    gt_df = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    gt_df = gt_df.with_columns(pl.col("matched_entity_ids").str.split(",").alias("candidate_entity_id")).explode("candidate_entity_id")
    gt_df = gt_df.with_columns(pl.col("candidate_entity_id").str.strip_chars())
    gt_df = gt_df.filter(pl.col("candidate_entity_id") != "")
    
    gt_tune = set(zip(
        gt_df.filter(pl.col("source1_entity_id").is_in(tune_s1_ids))["source1_entity_id"].to_list(),
        gt_df.filter(pl.col("source1_entity_id").is_in(tune_s1_ids))["candidate_entity_id"].to_list()
    ))
    gt_eval = set(zip(
        gt_df.filter(pl.col("source1_entity_id").is_in(eval_s1_ids))["source1_entity_id"].to_list(),
        gt_df.filter(pl.col("source1_entity_id").is_in(eval_s1_ids))["candidate_entity_id"].to_list()
    ))
    
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    s23_lf = pl.concat([frames["train_source2"].lazy(), frames["train_source3"].lazy()], how="diagonal")
    
    s1_tune = s1.filter(pl.col("entity_id").is_in(tune_s1_ids))
    s1_eval = s1.filter(pl.col("entity_id").is_in(eval_s1_ids))
    
    # We need the corpus mapping for TF-IDF
    corpus_cache_path = "work/task06_candidate_optimization/corpus_eids.npy"
    os.makedirs("work/task06_candidate_optimization", exist_ok=True)
    if os.path.exists(corpus_cache_path):
        print("Loading cached corpus entity IDs...")
        corpus_eids = np.load(corpus_cache_path).tolist()
    else:
        print("Generating deterministic corpus entity IDs (S2 + S3)...")
        s2_eids = frames["train_source2"].select("entity_id").collect()["entity_id"].to_list()
        s3_eids = frames["train_source3"].select("entity_id").collect()["entity_id"].to_list()
        
        print(f"S2 count: {len(s2_eids)}")
        print(f"S3 count: {len(s3_eids)}")
        
        corpus_eids = s2_eids + s3_eids
        print(f"Total Corpus ID count: {len(corpus_eids)}")
        
        assert len(s2_eids) == 5034616, f"Expected 5034616 S2 rows, got {len(s2_eids)}"
        assert len(s3_eids) == 5285603, f"Expected 5285603 S3 rows, got {len(s3_eids)}"
        assert len(corpus_eids) == 10320219, f"Expected 10320219 total rows, got {len(corpus_eids)}"
        assert len(set(corpus_eids)) == 10320219, "Corpus entity IDs contain duplicates!"
        
        print("Caching corpus entity IDs...")
        np.save(corpus_cache_path, np.array(corpus_eids))

    # 1. Generate Deterministic Blocks for Tuning Set
    print("Executing Deterministic Blocks on Tuning Set...")
    c_name = exact_name_blocking(s1_tune.lazy(), s23_lf)
    c_addr = exact_address_blocking(s1_tune.lazy(), s23_lf)
    c_ph = postal_house_blocking(s1_tune.lazy(), s23_lf)
    c_cs = cross_script_name_blocking(s1_tune.lazy(), s23_lf)
    c_abbr = abbreviation_blocking(s1_tune.lazy(), s23_lf)
    
    c_new_ph_name = postal_house_partial_name_blocking(s1_tune.lazy(), s23_lf)
    
    det_base = [c_name, c_addr, c_ph, c_cs, c_abbr]
    
    # 2. Run TF-IDF Experiments on Tuning Set
    print("Running TF-IDF Experiments on Tuning Set...")
    # Baseline K values
    tf_name_50, _ = run_tfidf_chunked_local(s1_tune, "latin_name", "word", (1,1), 50, "tfidf_name", "name_word", corpus_eids, 'tune')
    tf_addr_20, _ = run_tfidf_chunked_local(s1_tune, "latin_address", "word", (1,2), 20, "tfidf_addr", "address_word", corpus_eids, 'tune')
    tf_char_20, _ = run_tfidf_chunked_local(s1_tune, "latin_name", "char_wb", (3,4), 20, "tfidf_char", "name_char", corpus_eids, 'tune')
    
    # Expanded K values
    tf_name_75, _ = run_tfidf_chunked_local(s1_tune, "latin_name", "word", (1,1), 75, "tfidf_name", "name_word", corpus_eids, 'tune')
    tf_addr_30, _ = run_tfidf_chunked_local(s1_tune, "latin_address", "word", (1,2), 30, "tfidf_addr", "address_word", corpus_eids, 'tune')
    tf_char_30, _ = run_tfidf_chunked_local(s1_tune, "latin_name", "char_wb", (3,4), 30, "tfidf_char", "name_char", corpus_eids, 'tune')
    
    # Define Configurations
    configs = {
        "Baseline": {
            "blocks": det_base + [tf_name_50, tf_addr_20, tf_char_20]
        },
        "Exp_Name_K75": {
            "blocks": det_base + [tf_name_75, tf_addr_20, tf_char_20]
        },
        "Exp_Addr_K30": {
            "blocks": det_base + [tf_name_50, tf_addr_30, tf_char_20]
        },
        "Exp_Char_K30": {
            "blocks": det_base + [tf_name_50, tf_addr_20, tf_char_30]
        },
        "Exp_NewBlock_PH_Name": {
            "blocks": det_base + [c_new_ph_name, tf_name_50, tf_addr_20, tf_char_20]
        },
        "Combined_Best": {
            "blocks": det_base + [c_new_ph_name, tf_name_75, tf_addr_30, tf_char_30] # Will refine later
        }
    }
    
    results = {}
    
    def evaluate_config(cands_list, gt, s1_ids, name):
        out_dir = "work/task06_candidate_optimization/cache"
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(out_dir, f"{name}.parquet")
        
        if os.path.exists(parquet_path):
            print(f"Loading cached merged candidates for {name}...")
            merged = pl.read_parquet(parquet_path)
        else:
            merged = union_candidates(cands_list)
            merged.write_parquet(parquet_path)
            
        cand_pairs = set(zip(merged["source1_entity_id"].to_list(), merged["candidate_entity_id"].to_list()))
        retrieved = gt & cand_pairs
        recall = len(retrieved) / len(gt) if len(gt) > 0 else 0
        
        # We need counts per S1, even if 0
        counts_dict = {sid: 0 for sid in s1_ids}
        for row in merged.iter_rows(named=True):
            counts_dict[row["source1_entity_id"]] += 1
        counts = list(counts_dict.values())
        
        s1_metrics = []
        s1_recalls = []
        for sid in s1_ids:
            gt_s = {x[1] for x in gt if x[0] == sid}
            cands_s = {x[1] for x in cand_pairs if x[0] == sid}
            ret_s = gt_s & cands_s
            tp = len(ret_s)
            fn = len(gt_s) - tp
            fp = 0
            prec = tp / (tp + fp) if (tp+fp)>0 else (1.0 if tp==0 and fn==0 else 0.0)
            rec = tp / (tp + fn) if (tp+fn)>0 else (1.0 if tp==0 else 0.0)
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (prec+rec)>0 else 0.0
            s1_metrics.append(f05)
            s1_recalls.append(rec)
            
        return {
            "pair_recall": recall,
            "macro_recall": np.mean(s1_recalls),
            "oracle_macro_f05": np.mean(s1_metrics),
            "avg_cands": np.mean(counts) if counts else 0,
            "median_cands": np.median(counts) if counts else 0,
            "max_cands": np.max(counts) if counts else 0,
            "total_cands": len(cand_pairs)
        }
        
    baseline_metrics = None
    partial_json_path = "work/task06_candidate_optimization/cache/candidate_comparison_partial.json"
    os.makedirs(os.path.dirname(partial_json_path), exist_ok=True)

    def _atomic_json_dump(data, path):
        """Atomic write: write to .tmp then os.replace() to prevent truncation on crash."""
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=numpy_encoder)
        os.replace(tmp, path)

    if os.path.exists(partial_json_path):
        try:
            with open(partial_json_path, "r") as f:
                results = json.load(f)
            if "Baseline" in results:
                baseline_metrics = results["Baseline"]
            print(f"Resumed from partial checkpoint: {list(results.keys())}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"WARNING: Corrupted partial JSON ({e}). Starting fresh.")
            results = {}
    else:
        results = {}
        
    for name, conf in configs.items():
        if name == "Combined_Best": continue
        if name in results:
            print(f"Skipping {name}, already evaluated.")
            if name == "Baseline" and baseline_metrics is None:
                baseline_metrics = results["Baseline"]
            continue
            
        print(f"Evaluating {name}...")
        metrics = evaluate_config(conf["blocks"], gt_tune, tune_s1_ids, name)
        results[name] = metrics
        if name == "Baseline":
            baseline_metrics = metrics
        else:
            inc = ((metrics["total_cands"] - baseline_metrics["total_cands"]) / baseline_metrics["total_cands"]) * 100
            results[name]["cand_increase_pct"] = inc
            
        _atomic_json_dump(results, partial_json_path)
            
    # Challenge-aware selection logic
    best_combo_blocks = det_base + [tf_name_50, tf_addr_20, tf_char_20] # start with baseline
    combo_name = "Combined_Best"
    print("\nEvaluating Combined Config...")
    
    if results["Exp_Name_K75"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_Name_K75"]["cand_increase_pct"] < 20:
        best_combo_blocks[-3] = tf_name_75
    if results["Exp_Addr_K30"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_Addr_K30"]["cand_increase_pct"] < 20:
        best_combo_blocks[-2] = tf_addr_30
    if results["Exp_Char_K30"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_Char_K30"]["cand_increase_pct"] < 20:
        best_combo_blocks[-1] = tf_char_30
    if results["Exp_NewBlock_PH_Name"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_NewBlock_PH_Name"]["cand_increase_pct"] < 20:
        best_combo_blocks.append(c_new_ph_name)
        
    if combo_name in results:
        print(f"Skipping {combo_name}, already evaluated.")
        metrics = results[combo_name]
    else:
        metrics = evaluate_config(best_combo_blocks, gt_tune, tune_s1_ids, combo_name)
        inc = ((metrics["total_cands"] - baseline_metrics["total_cands"]) / baseline_metrics["total_cands"]) * 100
        metrics["cand_increase_pct"] = inc
        results[combo_name] = metrics
        _atomic_json_dump(results, partial_json_path)
    
    os.makedirs("work/task06_candidate_optimization", exist_ok=True)
    _atomic_json_dump(results, "work/task06_candidate_optimization/candidate_comparison.json")
        
    print("\n--- TUNING RESULTS ---")
    for name, mets in results.items():
        print(f"[{name}] Recall: {mets['pair_recall']:.4f}, Oracle F0.5: {mets['oracle_macro_f05']:.4f}, Avg Cands: {mets['avg_cands']:.1f}")

    # For Kaggle Execution, we would normally select the best config here and run on Evaluation set.
    # To save Kaggle time, we will just prepare the Combined_Best and run it on Evaluation set as a diagnostic.
    print("\n--- PHASE 6: VALIDATION ON UNTOUCHED EVAL ---")
    
    eval_blocks = [
        exact_name_blocking(s1_eval.lazy(), s23_lf),
        exact_address_blocking(s1_eval.lazy(), s23_lf),
        postal_house_blocking(s1_eval.lazy(), s23_lf),
        cross_script_name_blocking(s1_eval.lazy(), s23_lf),
        abbreviation_blocking(s1_eval.lazy(), s23_lf)
    ]
    
    tf_name_k = 75 if results["Exp_Name_K75"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_Name_K75"]["cand_increase_pct"] < 20 else 50
    tf_addr_k = 30 if results["Exp_Addr_K30"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_Addr_K30"]["cand_increase_pct"] < 20 else 20
    tf_char_k = 30 if results["Exp_Char_K30"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_Char_K30"]["cand_increase_pct"] < 20 else 20
    
    tf_name_eval, _ = run_tfidf_chunked_local(s1_eval, "latin_name", "word", (1,1), tf_name_k, "tfidf_name", "name_word", corpus_eids, 'eval')
    tf_addr_eval, _ = run_tfidf_chunked_local(s1_eval, "latin_address", "word", (1,2), tf_addr_k, "tfidf_addr", "address_word", corpus_eids, 'eval')
    tf_char_eval, _ = run_tfidf_chunked_local(s1_eval, "latin_name", "char_wb", (3,4), tf_char_k, "tfidf_char", "name_char", corpus_eids, 'eval')
    
    eval_blocks.extend([tf_name_eval, tf_addr_eval, tf_char_eval])
    
    if results["Exp_NewBlock_PH_Name"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_NewBlock_PH_Name"]["cand_increase_pct"] < 20:
        c_new_ph_name_eval = postal_house_partial_name_blocking(s1_eval.lazy(), s23_lf)
        eval_blocks.append(c_new_ph_name_eval)
        
    eval_metrics = evaluate_config(eval_blocks, gt_eval, eval_s1_ids)
    
    print(f"[Combined_Best Untouched Eval] Pair Recall: {eval_metrics['pair_recall']:.4f}, Macro Recall: {eval_metrics['macro_recall']:.4f}, Oracle F0.5: {eval_metrics['oracle_macro_f05']:.4f}, Avg Cands: {eval_metrics['avg_cands']:.1f}")
    
    out_config = {
        "name_tfidf_k": tf_name_k,
        "addr_tfidf_k": tf_addr_k,
        "char_tfidf_k": tf_char_k,
        "use_postal_house_fuzzy_name": results["Exp_NewBlock_PH_Name"]["pair_recall"] - baseline_metrics["pair_recall"] > 0.001 and results["Exp_NewBlock_PH_Name"]["cand_increase_pct"] < 20
    }
    _atomic_json_dump(out_config, "work/task06_candidate_optimization/selected_candidate_config.json")
    
    # Re-generate optimized candidates for both train and val and save
    print("\n--- PHASE 7: GENERATE OPTIMIZED CANDIDATES ---")
    s1_all = s1.filter(pl.col("entity_id").is_in(train_s1_ids + val_s1_ids))
    
    all_blocks = [
        exact_name_blocking(s1_all.lazy(), s23_lf),
        exact_address_blocking(s1_all.lazy(), s23_lf),
        postal_house_blocking(s1_all.lazy(), s23_lf),
        cross_script_name_blocking(s1_all.lazy(), s23_lf),
        abbreviation_blocking(s1_all.lazy(), s23_lf)
    ]
    
    tf_name_all, _ = run_tfidf_chunked_local(s1_all, "latin_name", "word", (1,1), tf_name_k, "tfidf_name", "name_word", corpus_eids, 'all')
    tf_addr_all, _ = run_tfidf_chunked_local(s1_all, "latin_address", "word", (1,2), tf_addr_k, "tfidf_addr", "address_word", corpus_eids, 'all')
    tf_char_all, _ = run_tfidf_chunked_local(s1_all, "latin_name", "char_wb", (3,4), tf_char_k, "tfidf_char", "name_char", corpus_eids, 'all')
    
    all_blocks.extend([tf_name_all, tf_addr_all, tf_char_all])
    
    if out_config["use_postal_house_fuzzy_name"]:
        all_blocks.append(postal_house_partial_name_blocking(s1_all.lazy(), s23_lf))
        
    merged_all = union_candidates(all_blocks)
    merged_all.write_parquet("work/task06_candidate_optimization/candidate_pairs_optimized.parquet")

if __name__ == "__main__":
    run()
