import polars as pl

def cross_script_name_blocking(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame) -> pl.DataFrame:
    """Blocks on exact latin_name for cross-script bridging."""
    s1 = s1_lf.filter(pl.col("latin_name") != "")
    s23 = s2_s3_lf.filter(pl.col("latin_name") != "")
    
    # We only care if it's actually matching something that wasn't already caught by exact normalized name?
    # Actually, we can just grab everything and union it later.
    
    joined = s1.join(
        s23, 
        on="latin_name", 
        how="inner",
        suffix="_candidate"
    )
    
    cands = joined.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
        pl.lit("latin_name").alias("retrieval_channels")
    ]).unique()
    
    return cands.collect()
