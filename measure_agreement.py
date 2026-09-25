import polars as pl
from src.data.loader import load_and_convert_tsv

def measure_agreement():
    print("Loading datasets...")
    # Read ground truth
    gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    
    # Let's take a sample of 10,000 matches to keep it fast for profiling
    gt = gt.head(10000)
    
    # We need to expand matched_entity_ids (comma separated) into rows
    gt_exploded = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
    )
    
    s1 = load_and_convert_tsv("dataset/train/train_source1.tsv", "work/parquet/train/source1.parquet").collect()
    s2 = load_and_convert_tsv("dataset/train/train_source2.tsv", "work/parquet/train/source2.parquet").collect()
    s3 = load_and_convert_tsv("dataset/train/train_source3.tsv", "work/parquet/train/source3.parquet").collect()
    
    # Combine S2 and S3 for lookup
    s2_s3 = pl.concat([s2, s3], how="diagonal")
    
    # Join S1 features
    joined = gt_exploded.join(s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
    
    # Join S2/S3 features
    joined = joined.join(s2_s3, left_on="matched_entity_ids", right_on="entity_id", how="inner", suffix="_right")
    
    total_pairs = joined.height
    print(f"Evaluated {total_pairs} true match pairs.")
    
    if total_pairs == 0:
        return
        
    def pct(count):
        return f"{(count / total_pairs) * 100:.2f}%"
        
    exact_raw_name = joined.filter(pl.col("business_name") == pl.col("business_name_right")).height
    exact_norm_name = joined.filter(pl.col("normalized_name") == pl.col("normalized_name_right")).height
    exact_latin_name = joined.filter(pl.col("latin_name") == pl.col("latin_name_right")).height
    
    exact_raw_addr = joined.filter(pl.col("business_address") == pl.col("business_address_right")).height
    exact_norm_addr = joined.filter(pl.col("normalized_address") == pl.col("normalized_address_right")).height
    
    exact_postal = joined.filter((pl.col("postal_code") != "") & (pl.col("postal_code") == pl.col("postal_code_right"))).height
    exact_house = joined.filter((pl.col("house_number") != "") & (pl.col("house_number") == pl.col("house_number_right"))).height
    
    print(f"Exact raw name agreement: {pct(exact_raw_name)}")
    print(f"Exact normalized name agreement: {pct(exact_norm_name)}")
    print(f"Exact Latin name agreement: {pct(exact_latin_name)}")
    
    print(f"Exact raw address agreement: {pct(exact_raw_addr)}")
    print(f"Exact normalized address agreement: {pct(exact_norm_addr)}")
    
    print(f"Exact postal agreement: {pct(exact_postal)}")
    print(f"Exact house-number agreement: {pct(exact_house)}")

if __name__ == "__main__":
    measure_agreement()
