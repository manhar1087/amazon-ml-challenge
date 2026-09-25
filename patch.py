import sys, json, os

lines = open('retrieval_benchmark.py').readlines()
new_lines = []
in_func = False

for line in lines:
    if line.startswith('def chunked_topk('):
        in_func = True
        new_lines.append(line)
        continue
    if in_func and line.startswith('for cfg in CHANNELS:'):
        in_func = False
    
    if not in_func:
        new_lines.append(line)

func_body = """    n_q = Q.shape[0]
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

"""

with open('retrieval_benchmark.py', 'w') as f:
    for line in new_lines:
        if line.startswith('def chunked_topk('):
            f.write(line)
            f.write(func_body)
        else:
            f.write(line)
