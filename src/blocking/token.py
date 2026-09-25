import polars as pl

def token_blocking(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame, min_shared_tokens: int = 2) -> pl.DataFrame:
    """
    Splits latin_name into tokens, ignores common tokens, and blocks on shared tokens.
    Returns pairs that share at least `min_shared_tokens`.
    """
    
    # 1. Tokenize
    # Extract unique words (length >= 3 to avoid noise like 'a', 'of')
    def tokenize(lf, id_col="entity_id"):
        return (
            lf.select([
                pl.col(id_col),
                pl.col("latin_name").str.split(" ")
            ])
            .explode("latin_name")
            .filter(pl.col("latin_name").str.len_chars() >= 3)
            .rename({"latin_name": "token"})
        )

    s1_tokens = tokenize(s1_lf)
    s23_tokens = tokenize(s2_s3_lf)
    
    # Find common tokens in S2/S3 to ignore them (Document Frequency > threshold)
    # If a word appears too often, it causes combinatorial explosions.
    token_counts = s23_tokens.group_by("token").len()
    rare_tokens = token_counts.filter(pl.col("len") < 500).select("token")
    
    # Filter tokens
    s1_tokens = s1_tokens.join(rare_tokens, on="token", how="inner")
    s23_tokens = s23_tokens.join(rare_tokens, on="token", how="inner")
    
    # 3. Join on exact token (Inverted index)
    joined = s1_tokens.join(
        s23_tokens,
        on="token",
        how="inner",
        suffix="_candidate"
    )
    
    # 4. Group by pairs and count shared tokens
    pair_counts = joined.group_by(["entity_id", "entity_id_candidate"]).len()
    
    # 5. Filter threshold
    valid_pairs = pair_counts.filter(pl.col("len") >= min_shared_tokens)
    
    cands = valid_pairs.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
        pl.lit(f"shared_{min_shared_tokens}_tokens").alias("retrieval_channels")
    ])
    
    return cands.collect()
