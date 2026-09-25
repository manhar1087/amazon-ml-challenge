import os, sys, json
import numpy as np
import scipy.sparse as sp

CSR_BASE = "incremental_csr_output"

def verify():
    print("="*70)
    print("FULL CORPUS INTEGRITY VERIFICATION")
    print("="*70)
    
    with open("full_build_manifest.json") as f:
        manifest = json.load(f)
        
    assert manifest["status"] == "SUCCESS"
    assert manifest["total_corpus_rows"] == 10_320_219
    
    channels = list(manifest["channels"].keys())
    
    for ch in channels:
        print(f"\nVerifying channel: {ch}")
        ch_meta = manifest["channels"][ch]
        
        c_dir = os.path.join(CSR_BASE, ch, "chunk_50000")
        chunk_folders = sorted([d for d in os.listdir(c_dir) if d.startswith("chunk_")])
        
        assert len(chunk_folders) == ch_meta["num_chunks"], "Chunk count mismatch"
        
        expected_start = 0
        total_nnz = 0
        total_rows = 0
        
        for d in chunk_folders:
            p = os.path.join(c_dir, d)
            with open(p + "/metadata.json") as f:
                meta = json.load(f)
                
            assert meta["global_row_start"] == expected_start, f"Contiguity error at {d}"
            
            # Load arrays to verify integrity
            try:
                data = np.load(p + "/data.npy")
                indices = np.load(p + "/indices.npy")
                indptr = np.load(p + "/indptr.npy")
                eids = np.load(p + "/entity_ids.npy")
            except Exception as e:
                print(f"FAILED to load chunk {d}: {e}")
                sys.exit(1)
                
            assert len(data) == len(indices) == meta["nnz"], f"NNZ mismatch in {d}"
            assert len(indptr) == meta["n_rows"] + 1, f"indptr mismatch in {d}"
            assert len(eids) == meta["n_rows"], f"entity_ids mismatch in {d}"
            
            expected_start += meta["n_rows"]
            total_rows += meta["n_rows"]
            total_nnz += len(data)
            
        assert total_rows == 10_320_219, f"Total rows mismatch: {total_rows}"
        assert total_nnz == ch_meta["total_nnz"], f"Total NNZ mismatch: {total_nnz}"
        
        print(f"  [PASS] {ch}: {total_rows} rows represented contiguously. NNZ matches manifest. All chunks loadable.")
        
    print("\nALL VERIFICATIONS PASSED.")

if __name__ == '__main__':
    verify()
