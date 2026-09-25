import numpy as np
import scipy.sparse as sp

from src.blocking.sparse_retrieval import get_deterministic_top_k

def test_chunked_merge():
    print("Testing independent chunks merged globally with deterministic ties...")
    K = 4
    n_q = 1
    best_scores  = np.full((n_q, K), -np.inf, dtype=np.float64)
    best_gidx    = np.full((n_q, K), -1,      dtype=np.int64)
    
    Q = sp.csr_matrix([[1.0]], dtype=np.float32)
    
    d1 = np.array([0.1, 0.5, 0.5, 0.9, 0.5], dtype=np.float32)
    C1 = sp.csr_matrix(d1.reshape(5, 1))
    
    d2 = np.array([0.5, 0.5, 0.9, 0.2, 0.8], dtype=np.float32)
    C2 = sp.csr_matrix(d2.reshape(5, 1))
    
    chunks = [(0, C1), (5, C2)]
    
    for g_start, C in chunks:
        n_chunk = C.shape[0]
        sim = (Q @ C.T).toarray()[0]
        
        lg = np.arange(n_chunk, dtype=np.int64) + g_start
        comb_s = np.concatenate([best_scores[0], sim])
        comb_g = np.concatenate([best_gidx[0], lg])
        
        valid = comb_g >= 0
        bs, bg = get_deterministic_top_k(comb_s[valid], comb_g[valid], K)
        
        best_scores[0, :len(bs)] = bs
        best_gidx[0, :len(bg)] = bg
        
    final_s, final_g = get_deterministic_top_k(best_scores[0], best_gidx[0], K)
    print(f"Final scores: {final_s}")
    print(f"Final global idxs: {final_g}")
    
    assert list(final_g) == [3, 7, 9, 1] # 0.9, 0.9, 0.8, 0.5 (from global 1)
    print("PASS: Cross-chunk independent processing correctly merges and ranks globally.")

if __name__ == '__main__':
    test_chunked_merge()
