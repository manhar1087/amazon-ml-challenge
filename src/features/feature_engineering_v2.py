import polars as pl
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz
import numpy as np

def jaro_winkler_sim(df, col1, col2):
    c1 = df[col1].to_list()
    c2 = df[col2].to_list()
    res = [JaroWinkler.normalized_similarity(str(x) if x else "", str(y) if y else "") if x is not None and y is not None else 0.0 for x, y in zip(c1, c2)]
    return pl.Series(res, dtype=pl.Float32)

def levenshtein_sim(df, col1, col2):
    c1 = df[col1].to_list()
    c2 = df[col2].to_list()
    res = [Levenshtein.normalized_similarity(str(x) if x else "", str(y) if y else "") if x is not None and y is not None else 0.0 for x, y in zip(c1, c2)]
    return pl.Series(res, dtype=pl.Float32)

def token_jaccard(df, col1, col2):
    c1 = df[col1].to_list()
    c2 = df[col2].to_list()
    res = []
    for x, y in zip(c1, c2):
        if x and y:
            s1 = set(str(x).split())
            s2 = set(str(y).split())
            u = len(s1 | s2)
            res.append(len(s1 & s2) / u if u > 0 else 0.0)
        else:
            res.append(0.0)
    return pl.Series(res, dtype=pl.Float32)

def token_sort_ratio(df, col1, col2):
    c1 = df[col1].to_list()
    c2 = df[col2].to_list()
    res = [fuzz.token_sort_ratio(str(x) if x else "", str(y) if y else "") / 100.0 if x is not None and y is not None else 0.0 for x, y in zip(c1, c2)]
    return pl.Series(res, dtype=pl.Float32)
    
def token_set_ratio(df, col1, col2):
    c1 = df[col1].to_list()
    c2 = df[col2].to_list()
    res = [fuzz.token_set_ratio(str(x) if x else "", str(y) if y else "") / 100.0 if x is not None and y is not None else 0.0 for x, y in zip(c1, c2)]
    return pl.Series(res, dtype=pl.Float32)

def build_features_v2(pairs_lf: pl.LazyFrame, s1_lf: pl.LazyFrame, s23_lf: pl.LazyFrame) -> pl.DataFrame:
    s1_cols = ["entity_id", "country", "business_name", "normalized_name", "latin_name", "business_address", "normalized_address", "postal_code", "house_number"]
    s23_cols = s1_cols.copy()
    s23_cols.append("source")
    
    df = pairs_lf.join(s1_lf.select(s1_cols), left_on="source1_entity_id", right_on="entity_id", how="left")
    df = df.join(s23_lf.select(s23_cols), left_on="candidate_entity_id", right_on="entity_id", how="left", suffix="_right")
    
    df = df.collect()
    
    df = df.with_columns([
        (pl.col("country") == "France").cast(pl.Float32).fill_null(0.0).alias("f_is_france"),
        (pl.col("country") == "US").cast(pl.Float32).fill_null(0.0).alias("f_is_us"),
        (pl.col("country") == "India").cast(pl.Float32).fill_null(0.0).alias("f_is_india"),
        (pl.col("source") == "S3").cast(pl.Float32).fill_null(0.0).alias("f_is_s3"),
    ])
    
    df = df.with_columns([
        (pl.col("normalized_name") == pl.col("normalized_name_right")).cast(pl.Float32).fill_null(0.0).alias("f_name_exact"),
        (pl.col("latin_name") == pl.col("latin_name_right")).cast(pl.Float32).fill_null(0.0).alias("f_latin_exact"),
        (pl.col("normalized_address") == pl.col("normalized_address_right")).cast(pl.Float32).fill_null(0.0).alias("f_addr_exact"),
        (pl.col("postal_code") == pl.col("postal_code_right")).cast(pl.Float32).fill_null(0.0).alias("f_postal_match"),
        (pl.col("house_number") == pl.col("house_number_right")).cast(pl.Float32).fill_null(0.0).alias("f_house_match"),
        (pl.col("business_address") == "").cast(pl.Float32).fill_null(0.0).alias("f_missing_addr_s1"),
        (pl.col("business_address_right") == "").cast(pl.Float32).fill_null(0.0).alias("f_missing_addr_s23")
    ])
    
    # Conflict features
    df = df.with_columns([
        ((pl.col("postal_code").is_not_null()) & (pl.col("postal_code_right").is_not_null()) & (pl.col("postal_code") != pl.col("postal_code_right"))).cast(pl.Float32).alias("f_postal_conflict"),
        ((pl.col("house_number").is_not_null()) & (pl.col("house_number_right").is_not_null()) & (pl.col("house_number") != pl.col("house_number_right"))).cast(pl.Float32).alias("f_house_conflict")
    ])
    
    df = df.with_columns([
        (pl.col("latin_name").str.len_chars() / pl.col("latin_name_right").str.len_chars().clip(lower_bound=1)).cast(pl.Float32).fill_null(0.0).alias("f_name_len_ratio"),
        (pl.col("normalized_address").str.len_chars() / pl.col("normalized_address_right").str.len_chars().clip(lower_bound=1)).cast(pl.Float32).fill_null(0.0).alias("f_addr_len_ratio")
    ])
    
    # String metrics
    df = df.with_columns([
        jaro_winkler_sim(df, "latin_name", "latin_name_right").alias("f_name_jaro"),
        levenshtein_sim(df, "latin_name", "latin_name_right").alias("f_name_lev"),
        token_jaccard(df, "latin_name", "latin_name_right").alias("f_name_jaccard"),
        token_sort_ratio(df, "latin_name", "latin_name_right").alias("f_name_sort_ratio"),
        token_set_ratio(df, "latin_name", "latin_name_right").alias("f_name_set_ratio"),
        
        jaro_winkler_sim(df, "normalized_address", "normalized_address_right").alias("f_addr_jaro"),
        levenshtein_sim(df, "normalized_address", "normalized_address_right").alias("f_addr_lev"),
        token_jaccard(df, "normalized_address", "normalized_address_right").alias("f_addr_jaccard"),
        token_set_ratio(df, "normalized_address", "normalized_address_right").alias("f_addr_set_ratio")
    ])
    
    df = df.with_columns([
        (pl.col("f_name_jaro") * pl.col("f_addr_jaro")).alias("f_cross_name_addr_jaro"),
        ((pl.col("f_name_jaro") > 0.9) & (pl.col("f_addr_jaro") > 0.8)).cast(pl.Float32).alias("f_cross_strong_both"),
        ((pl.col("f_name_sort_ratio") > 0.9) & (pl.col("f_postal_conflict") == 0.0)).cast(pl.Float32).alias("f_strong_name_no_postal_conflict")
    ])
    
    if "retrieval_channels" in df.columns:
        channels = ["exact_name", "exact_addr", "structural", "cross_script", "abbr", "tfidf_name", "tfidf_addr", "tfidf_char"]
        for c in channels:
            df = df.with_columns(
                pl.col("retrieval_channels").str.contains(c).cast(pl.Float32).fill_null(0.0).alias(f"f_retr_{c}")
            )
        df = df.with_columns(
            pl.col("retrieval_channels").str.count_matches(r"\|").add(1).cast(pl.Float32).fill_null(0.0).alias("f_retr_count")
        )
    else:
        channels = ["exact_name", "exact_addr", "structural", "cross_script", "abbr", "tfidf_name", "tfidf_addr", "tfidf_char"]
        for c in channels:
            df = df.with_columns(pl.lit(0.0).alias(f"f_retr_{c}"))
        df = df.with_columns(pl.lit(1.0).alias("f_retr_count"))
        
    cand_counts = df.group_by("source1_entity_id").agg(pl.len().alias("f_ctx_cand_count"))
    df = df.join(cand_counts, on="source1_entity_id", how="left").with_columns(pl.col("f_ctx_cand_count").cast(pl.Float32))
    
    return df
