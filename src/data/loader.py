import polars as pl
import os
from .normalization import add_normalized_columns

def load_and_convert_tsv(tsv_path: str, parquet_path: str, force: bool = False) -> pl.LazyFrame:
    """
    Converts a TSV file to Parquet format for fast subsequent loads.
    Applies normalization pipeline during conversion.
    Returns a LazyFrame pointing to the Parquet file.
    """
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    
    if force or not os.path.exists(parquet_path):
        print(f"Converting {tsv_path} to {parquet_path}...")
        # Read raw TSV
        lf = pl.scan_csv(
            tsv_path, 
            separator="\t",
            infer_schema_length=10000,
            null_values=["", "NaN", "null"],
            missing_utf8_is_empty_string=False
        )
        
        # Apply normalization
        lf = add_normalized_columns(lf)
        
        # Write to Parquet
        # Collect executes the graph
        df = lf.collect(streaming=True)
        df.write_parquet(parquet_path)
        print(f"Finished writing {parquet_path}")
        
    return pl.scan_parquet(parquet_path)

def load_all_data(work_dir: str = "work", data_dir: str = "dataset") -> dict:
    """
    Loads all train and test sets into a dictionary of LazyFrames,
    converting to Parquet if necessary.
    """
    splits = ["train", "test"]
    sources = ["source1", "source2", "source3"]
    
    frames = {}
    for split in splits:
        for source in sources:
            tsv_file = f"{split}_{source}.tsv"
            # Train ground truth doesn't need normalization
            
            tsv_path = os.path.join(data_dir, split, tsv_file)
            parquet_path = os.path.join(work_dir, "parquet", split, f"{source}.parquet")
            
            if os.path.exists(tsv_path):
                frames[f"{split}_{source}"] = load_and_convert_tsv(tsv_path, parquet_path)
                
    # Also copy over train_ground_truth as parquet
    gt_tsv = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    gt_pq = os.path.join(work_dir, "parquet", "train", "train_ground_truth.parquet")
    if os.path.exists(gt_tsv):
        if not os.path.exists(gt_pq):
            pl.scan_csv(gt_tsv, separator="\t").collect().write_parquet(gt_pq)
        frames["train_ground_truth"] = pl.scan_parquet(gt_pq)
        
    return frames
