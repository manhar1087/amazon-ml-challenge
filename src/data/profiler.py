import polars as pl
from typing import Dict, Any

def profile_dataframe(df: pl.DataFrame, split_name: str) -> Dict[str, Any]:
    """
    Calculates detailed statistics for a normalized DataFrame.
    """
    total_rows = df.height
    if total_rows == 0:
        return {"error": "Empty dataframe"}
        
    # Missing addresses
    missing_address_count = df.filter(pl.col("business_address") == "").height
    missing_address_rate = missing_address_count / total_rows
    
    # Extraction rates
    postal_extracted = df.filter(pl.col("postal_code") != "").height
    house_extracted = df.filter(pl.col("house_number") != "").height
    
    # Exact normalized duplicates
    # Group by normalized name & address, filter count > 1
    exact_duplicates = df.group_by(["normalized_name", "normalized_address"]).len().filter(pl.col("len") > 1).select(pl.sum("len")).item()
    exact_duplicates = exact_duplicates if exact_duplicates else 0
    
    # Name and Address lengths (characters of raw)
    # Handle nulls safely
    raw_name_len = df.select(pl.col("business_name").fill_null("").str.len_chars().mean()).item()
    raw_addr_len = df.select(pl.col("business_address").fill_null("").str.len_chars().mean()).item()
    
    # Collision sizes for normalized name
    name_collisions = df.group_by("normalized_name").len()
    max_name_collision = name_collisions.select(pl.col("len").max()).item()
    avg_name_collision = name_collisions.select(pl.col("len").mean()).item()
    
    # Mixed/Indic script presence (basic heuristic: difference between raw length and unidecode length, 
    # or just checking if normalized_name != latin_name after stripping spaces)
    # A more precise way is to count rows where latin_name differs from normalized_name for reasons other than space
    diff_script = df.filter(
        (pl.col("normalized_name") != pl.col("latin_name")) |
        (pl.col("normalized_address") != pl.col("latin_address"))
    ).height
    
    stats = {
        "split": split_name,
        "total_rows": total_rows,
        "missing_address_rate": missing_address_rate,
        "postal_extraction_rate": postal_extracted / total_rows,
        "house_extraction_rate": house_extracted / total_rows,
        "normalized_exact_duplicate_rate": exact_duplicates / total_rows,
        "avg_raw_name_length": raw_name_len,
        "avg_raw_address_length": raw_addr_len,
        "max_name_collision": max_name_collision,
        "avg_name_collision": avg_name_collision,
        "cross_script_alteration_rate": diff_script / total_rows
    }
    return stats

def run_profiling(frames: dict, country_filter: str = None) -> dict:
    """
    Runs profiling over loaded frames. Use collect() directly since profiling needs materialized data.
    To avoid OOM, use a sample if the data is massive, but for 2M rows polars handles it in memory.
    """
    results = {}
    for name, lf in frames.items():
        if "ground_truth" in name:
            continue
            
        if country_filter:
            lf = lf.filter(pl.col("country") == country_filter)
            
        df = lf.collect()
        results[name] = profile_dataframe(df, name)
        
    return results
