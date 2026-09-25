import polars as pl

def exact_name_blocking(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame) -> pl.DataFrame:
    """Blocks on exact normalized_name."""
    # Filter out empty names
    s1 = s1_lf.filter(pl.col("normalized_name") != "")
    s23 = s2_s3_lf.filter(pl.col("normalized_name") != "")
    
    joined = s1.join(
        s23, 
        on="normalized_name", 
        how="inner",
        suffix="_candidate"
    )
    
    cands = joined.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
        pl.lit("exact_name").alias("retrieval_channels")
    ]).unique()
    
    return cands.collect()

def exact_address_blocking(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame) -> pl.DataFrame:
    """Blocks on exact normalized_address."""
    s1 = s1_lf.filter(pl.col("normalized_address") != "")
    s23 = s2_s3_lf.filter(pl.col("normalized_address") != "")
    
    joined = s1.join(
        s23, 
        on="normalized_address", 
        how="inner",
        suffix="_candidate"
    )
    
    cands = joined.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
        pl.lit("exact_address").alias("retrieval_channels")
    ]).unique()
    
    return cands.collect()
