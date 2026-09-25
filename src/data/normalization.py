import polars as pl
from unidecode import unidecode
import re

def latinize_series(s: pl.Series) -> pl.Series:
    """Applies unidecode to convert text to ASCII/Latin representation."""
    # Handle nulls gracefully by mapping to empty string or returning null
    return s.fill_null("").map_elements(lambda x: unidecode(x) if x else "", return_dtype=pl.String)

def normalize_text_expr(col_name: str, is_name: bool = False) -> pl.Expr:
    """
    Returns a Polars expression to normalize a text column.
    Preserves Unicode letters/numbers.
    """
    expr = (
        pl.col(col_name).fill_null("")
        .str.to_lowercase()
        # Replace punctuation (anything not a letter, number, mark, or whitespace) with space
        .str.replace_all(r"[^\p{L}\p{N}\p{M}\s]", " ")
        # Collapse multiple spaces
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
    
    if is_name:
        # Legal suffix normalization (conservative)
        expr = (
            expr
            .str.replace_all(r"\b(?:private limited|pvt ltd)\b", "pvt ltd")
            .str.replace_all(r"\b(?:limited|ltd)\b", "ltd")
            .str.replace_all(r"\b(?:corporation|corp)\b", "corp")
            .str.replace_all(r"\b(?:llc|l l c)\b", "llc")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
    return expr

def add_normalized_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Takes a LazyFrame with raw data and adds:
    - normalized_name (Unicode preserved)
    - latin_name (Transliterated)
    - normalized_address (Unicode preserved)
    - latin_address (Transliterated)
    - postal_code
    - house_number
    - source (S1, S2, S3)
    """
    
    # Basic unicode-preserving normalization
    lf = lf.with_columns([
        normalize_text_expr("business_name", is_name=True).alias("normalized_name"),
        normalize_text_expr("business_address", is_name=False).alias("normalized_address"),
        pl.col("entity_id").str.slice(0, 2).alias("source")
    ])
    
    # Transliteration / Latin normalization
    lf = lf.with_columns([
        pl.col("normalized_name").map_batches(latinize_series).alias("latin_name"),
        pl.col("normalized_address").map_batches(latinize_series).alias("latin_address")
    ])
    
    # Address Structural Extraction
    # Postal code: US (5 or 5+4), India (6), France (5)
    # House number: leading digits followed by optional letter
    lf = lf.with_columns([
        pl.col("business_address").str.extract(r".*\b(\d{5}(?:-\d{4})?|\d{6})\b", 1).fill_null("").alias("postal_code"),
        pl.col("business_address").str.extract(r"^\s*(\d+[a-zA-Z]?)\b", 1).fill_null("").alias("house_number")
    ])
    
    return lf
