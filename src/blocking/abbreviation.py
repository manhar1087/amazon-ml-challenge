import polars as pl

def abbreviation_blocking(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Blocks S1 on S23 by comparing exact names with token initials (acronyms).
    Example: 'International Business Machines' -> 'ibm'
    """
    # Create acronym column by taking the first character of each word in latin_name
    def add_acronym(lf):
        return lf.with_columns([
            pl.col("latin_name").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("acronym")
        ])
        
    s1 = add_acronym(s1_lf).filter(pl.col("acronym").str.len_chars() >= 3)
    s23 = add_acronym(s2_s3_lf).filter(pl.col("acronym").str.len_chars() >= 3)
    
    # Block 1: S1 acronym == S23 exact name
    b1 = s1.join(
        s23.filter(pl.col("latin_name") != ""),
        left_on="acronym",
        right_on="latin_name",
        how="inner",
        suffix="_candidate"
    )
    
    # Block 2: S1 exact name == S23 acronym
    b2 = s1.filter(pl.col("latin_name") != "").join(
        s23,
        left_on="latin_name",
        right_on="acronym",
        how="inner",
        suffix="_candidate"
    )
    
    # Select columns
    def format_cands(df, name):
        return df.select([
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("entity_id_candidate").alias("candidate_entity_id"),
            pl.lit(name).alias("retrieval_channels")
        ]).unique()

    c1 = format_cands(b1, "acronym_s1_to_name")
    c2 = format_cands(b2, "name_to_acronym_s23")
    
    merged = pl.concat([c1, c2]).unique(subset=["source1_entity_id", "candidate_entity_id"])
    return merged.collect()
