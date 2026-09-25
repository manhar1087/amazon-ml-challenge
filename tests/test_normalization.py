import pytest
import polars as pl
from src.data.normalization import add_normalized_columns

@pytest.fixture
def sample_data():
    return pl.DataFrame({
        "entity_id": ["S1-001", "S2-002", "S3-003", "S1-004", "S2-005", "S2-006", "S2-007"],
        "business_name": [
            "Orelee's Barbershop!", # Punctuation
            "PRIME MONEY PRIVATE LIMITED", # Case & Suffix
            "B+ Retail Inc.", # Punctuation
            "Christ Chapel", # Standard
            "Consulting Nyasa Pvt Ltd", # Abbreviation
            None, # Missing name
            "राम मार्केटिंग" # Indic Script
        ],
        "business_address": [
            "1795 Westchester Drive, High Point, NC", # Standard US
            "17560 Ellis Road, Tahlequah, OK 74464", # US with ZIP
            "797, Lake Town Block A, Kolkata, 700089", # India with PIN
            None, # Missing address
            "2505, Tower 1, Mumbai", # No house number strictly at front
            "Opp Railway Station, Delhi", # Landmark
            "संजय नगर, 110001" # Indic + PIN
        ],
        "country": ["US", "US", "India", "US", "India", "India", "India"]
    }).lazy()

def test_normalization_pipeline(sample_data):
    result = add_normalized_columns(sample_data).collect()
    
    # 1. Source Identification
    assert result.filter(pl.col("entity_id") == "S1-001")["source"][0] == "S1"
    
    # 2. Case and Punctuation Normalization
    name1 = result.filter(pl.col("entity_id") == "S1-001")["normalized_name"][0]
    assert name1 == "orelee s barbershop" # Punctuation replaced by space
    
    # 3. Legal Suffix Normalization
    name2 = result.filter(pl.col("entity_id") == "S2-002")["normalized_name"][0]
    assert name2 == "prime money pvt ltd"
    
    # 4. Missing value handling
    name_missing = result.filter(pl.col("entity_id") == "S2-006")["normalized_name"][0]
    addr_missing = result.filter(pl.col("entity_id") == "S1-004")["normalized_address"][0]
    assert name_missing == ""
    assert addr_missing == ""
    
    # 5. Postal extraction
    zip_us = result.filter(pl.col("entity_id") == "S2-002")["postal_code"][0]
    pin_in = result.filter(pl.col("entity_id") == "S3-003")["postal_code"][0]
    assert zip_us == "74464"
    assert pin_in == "700089"
    
    # 6. House number extraction
    hn_us = result.filter(pl.col("entity_id") == "S1-001")["house_number"][0]
    assert hn_us == "1795"
    
    # 7. Cross-script representation (Unicode preservation vs Latin)
    indic_row = result.filter(pl.col("entity_id") == "S2-007")
    norm_name = indic_row["normalized_name"][0]
    latin_name = indic_row["latin_name"][0]
    
    assert "राम" in norm_name # Unicode preserved
    assert "raam" in latin_name.lower() # Transliterated to latin
    
    # Determinism: running twice yields identical results
    result2 = add_normalized_columns(sample_data).collect()
    assert result.equals(result2)

def test_accented_characters():
    df = pl.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Café Françoise"],
        "business_address": ["L'Oréal Paris"],
        "country": ["France"]
    }).lazy()
    
    res = add_normalized_columns(df).collect()
    # Unicode preserved in normalized
    assert res["normalized_name"][0] == "café françoise"
    # Unidecode translates Accents
    assert res["latin_name"][0] == "cafe francoise"
