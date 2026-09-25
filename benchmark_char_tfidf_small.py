import polars as pl
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import gc

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split

def run_small_benchmark():
    frames = load_all_data()
    s1 = frames["train_source1"].collect()
    s23 = pl.concat([frames["train_source2"].collect(), frames["train_source3"].collect()], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    # Create holdout IDs as before
    t_split, v_split = get_grouped_split(gt['source1_entity_id'].to_list())
    np.random.seed(999)
    strict_val_ids = np.random.choice(v_split, size=200, replace=False).tolist()
    s1_sub = s1.filter(pl.col("entity_id").is_in(strict_val_ids))

    col = "latin_name"
    analyzer = "char_wb"
    ngram_range = (3,4)
    min_df = 2
    max_df = 0.005
    k = 20

    X_c = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_df=max_df, dtype=np.float32).fit_transform(s23[col].to_list())
    X_q = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df, max_df=max_df, dtype=np.float32).fit(s23[col].to_list()).transform(s1_sub[col].to_list())
    s1_ids = s1_sub["entity_id"].to_numpy()
    s23_ids = s23["entity_id"].to_numpy()

    # Old method (dense) - may be ok for 200 queries
    t0 = time.time()
    sim_old = X_q.dot(X_c.T)
    old_time = time.time() - t0
    # extract top K
    old_matches = []
    for i in range(sim_old.shape[0]):
        data = sim_old.data[sim_old.indptr[i]:sim_old.indptr[i+1]]
        idx = sim_old.indices[sim_old.indptr[i]:sim_old.indptr[i+1]]
        if len(data) > k:
            top = np.argpartition(data, -k)[-k:]
            idx = idx[top]
        old_matches.append(set(s23_ids[idx]))

    # New method
    t1 = time.time()
    sim_new = sp_matmul_topn(X_q, X_c, top_n=k, sort=True)
    new_time = time.time() - t1
    new_matches = []
    for i in range(sim_new.shape[0]):
        idx = sim_new.indices[sim_new.indptr[i]:sim_new.indptr[i+1]]
        new_matches.append(set(s23_ids[idx]))

    # Compare sets
    exact = sum(1 for o,n in zip(old_matches,new_matches) if o==n)
    print(f"Exact set match count: {exact}/{len(old_matches)}")
    print(f"Old method time: {old_time:.2f}s, New method time: {new_time:.2f}s")

if __name__ == "__main__":
    run_small_benchmark()
