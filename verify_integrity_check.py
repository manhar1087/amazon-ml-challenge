import polars as pl
import os
import generate_task03_holdout as gen

def test_integrity():
    print("="*70)
    print("SMALL INTEGRITY TEST (Production Retrieval Path)")
    print("="*70)
    
    # 5 queries from S1
    s1 = pl.scan_parquet("work/parquet/train/source1.parquet").head(5).collect()
    
    # Dummy corpus_eids just for testing (we don't need all 10M to check candidate counts)
    # The actual retrieval will return global indices from 0 to 10.32M
    # So we need a dummy list of 10.32M strings to prevent index errors
    print("Building dummy corpus EIDs for index mapping...")
    dummy_eids = [f"E_{i}" for i in range(10320219)]
    
    channels = [
        {"name": "tfidf_name_k50", "col": "latin_name", "analyzer": "word", "ngram": (1,1), "k": 50},
        {"name": "tfidf_addr_k20", "col": "latin_address", "analyzer": "word", "ngram": (1,2), "k": 20},
        {"name": "tfidf_char_k20", "col": "latin_name", "analyzer": "char_wb", "ngram": (3,4), "k": 20},
    ]
    
    for cfg in channels:
        print(f"\nRunning test retrieval for {cfg['name']}...")
        df, vocab_size = gen.run_tfidf_chunked(
            s1, cfg["col"], cfg["analyzer"], cfg["ngram"], cfg["k"], 
            cfg["name"], dummy_eids, setup_dir="retrieval_setup"
        )
        
        cands_per_query = df.group_by("source1_entity_id").agg(pl.count("candidate_entity_id"))
        counts = cands_per_query["candidate_entity_id"].to_list()
        
        print(f"  Vocab Size: {vocab_size}")
        print(f"  Queries returned: {len(counts)}")
        print(f"  Candidates per query: {counts}")
        
        assert all(c == cfg["k"] for c in counts), f"Mismatch in K! Expected {cfg['k']}"
        print(f"  [PASS] {cfg['name']} returned exactly {cfg['k']} candidates per query.")
        
if __name__ == '__main__':
    test_integrity()
