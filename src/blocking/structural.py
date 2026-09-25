import polars as pl

def postal_house_blocking(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame) -> pl.DataFrame:
    """Blocks on both exact postal code and exact house number."""
    s1 = s1_lf.filter((pl.col("postal_code") != "") & (pl.col("house_number") != ""))
    s23 = s2_s3_lf.filter((pl.col("postal_code") != "") & (pl.col("house_number") != ""))
    
    joined = s1.join(
        s23, 
        on=["postal_code", "house_number"], 
        how="inner",
        suffix="_candidate"
    )
    
    cands = joined.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
        pl.lit("postal_house").alias("retrieval_channels")
    ]).unique()
    
    return cands.collect()
