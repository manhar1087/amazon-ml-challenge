import polars as pl

def ngram_signature_blocking(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Creates a character-level signature (e.g. sorted characters) for latin_name
    to recover typos, permutations, and spacing variations without pairwise N^2.
    """
    
    def add_signature(lf, id_col="entity_id"):
        # We'll use latin_name. Remove spaces, explode to chars, sort, and join back.
        # This creates an exact anagram signature: "mcdonalds" -> "acddlmnos"
        # Since doing this natively across rows can be tricky in lazy frames,
        # we can just use map_elements or regex tricks.
        # A simpler typo-tolerant block: First 3 and last 3 chars? No.
        # Let's use a simpler signature: remove vowels and spaces (Soundex-like)
        
        return lf.with_columns([
            pl.col("latin_name").str.replace_all(r"[aeiou\s]", "").alias("consonant_sig")
        ])

    s1 = add_signature(s1_lf).filter(pl.col("consonant_sig").str.len_chars() >= 4)
    s23 = add_signature(s2_s3_lf).filter(pl.col("consonant_sig").str.len_chars() >= 4)
    
    # We only want to join if the signature is somewhat rare to avoid explosions
    counts = s23.group_by("consonant_sig").len()
    rare_sigs = counts.filter(pl.col("len") < 100).select("consonant_sig")
    
    s1 = s1.join(rare_sigs, on="consonant_sig", how="inner")
    s23 = s23.join(rare_sigs, on="consonant_sig", how="inner")
    
    joined = s1.join(
        s23,
        on="consonant_sig",
        how="inner",
        suffix="_candidate"
    )
    
    cands = joined.select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_candidate").alias("candidate_entity_id"),
        pl.lit("consonant_signature").alias("retrieval_channels")
    ]).unique()
    
    return cands.collect()
