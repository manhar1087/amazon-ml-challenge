import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time
from src.data.loader import load_all_data

def test_sparse():
    frames = load_all_data()
    s1 = frames["train_source1"].collect().head(5000)
    s23 = frames["train_source2"].collect().head(100000) # testing on 100k
    
    print("Building Char N-Gram TF-IDF Vectorizer...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,4), min_df=2, max_df=0.05, dtype=np.float32)
    
    print("Fitting corpus...")
    X_s23 = vectorizer.fit_transform(s23["latin_name"].fill_null("").to_list())
    
    print("Transforming queries...")
    X_s1 = vectorizer.transform(s1["latin_name"].fill_null("").to_list())
    
    print(f"Vectors built in {time.time()-t0:.2f}s. Vocab size: {len(vectorizer.vocabulary_)}")
    
    print("Computing dot product chunk...")
    t0 = time.time()
    sim = X_s1.dot(X_s23.T)
    print(f"Dot product done in {time.time()-t0:.2f}s. Density: {sim.nnz / (sim.shape[0] * sim.shape[1]):.4f}")
    
    # Extract top k
    k = 5
    out_s1 = []
    out_s23 = []
    for row_idx in range(sim.shape[0]):
        row_data = sim.data[sim.indptr[row_idx]:sim.indptr[row_idx+1]]
        row_indices = sim.indices[sim.indptr[row_idx]:sim.indptr[row_idx+1]]
        if len(row_data) > 0:
            if len(row_data) > k:
                top_k_idx = np.argpartition(row_data, -k)[-k:]
                top_k_indices = row_indices[top_k_idx]
            else:
                top_k_indices = row_indices
            out_s1.extend([row_idx] * len(top_k_indices))
            
    print(f"Extraction done. Total pairs: {len(out_s1)}")

if __name__ == "__main__":
    test_sparse()
