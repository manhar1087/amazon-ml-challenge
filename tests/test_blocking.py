import pytest
import polars as pl
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from src.blocking.cross_script import cross_script_name_blocking
from src.blocking.token import token_blocking
from src.blocking.ngram import ngram_signature_blocking
from src.blocking.union import union_candidates

@pytest.fixture
def mock_s1():
    return pl.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "normalized_name": ["mcdonalds", "starbucks", "subway"],
        "normalized_address": ["123 main st", "456 oak ave", ""],
        "latin_name": ["mcdonalds", "starbucks", "subway"],
        "postal_code": ["10001", "20002", ""],
        "house_number": ["123", "456", ""]
    }).lazy()

@pytest.fixture
def mock_s23():
    return pl.DataFrame({
        "entity_id": ["S2-1", "S3-1", "S2-2", "S2-3"],
        "normalized_name": ["mcdonalds", "strabucks", "subway india", ""], # Typo in starbucks
        "normalized_address": ["123 main st", "456 oak ave", "123 main st", ""],
        "latin_name": ["mcdonalds", "strabucks", "subway india", ""],
        "postal_code": ["10001", "20002", "10001", ""],
        "house_number": ["123", "456", "123", ""]
    }).lazy()

def test_exact_name(mock_s1, mock_s23):
    res = exact_name_blocking(mock_s1, mock_s23)
    assert len(res) == 1
    assert res["source1_entity_id"][0] == "S1-1"
    assert res["candidate_entity_id"][0] == "S2-1"

def test_exact_address(mock_s1, mock_s23):
    res = exact_address_blocking(mock_s1, mock_s23)
    # S1-1 matches S2-1 and S2-2 on address
    assert len(res) == 3 # S1-1 -> S2-1, S1-1 -> S2-2, S1-2 -> S3-1

def test_structural_postal_house(mock_s1, mock_s23):
    res = postal_house_blocking(mock_s1, mock_s23)
    # S1-1 matches S2-1, S2-2 (same house and postal)
    # S1-2 matches S3-1
    assert len(res) == 3

def test_ngram_typo(mock_s1, mock_s23):
    res = ngram_signature_blocking(mock_s1, mock_s23)
    # Starbucks -> strabucks will match on consonant signature 'bckrsst'
    assert "S3-1" in res["candidate_entity_id"].to_list()

def test_union_candidates():
    b1 = pl.DataFrame({"source1_entity_id": ["1"], "candidate_entity_id": ["A"], "retrieval_channels": ["exact"]})
    b2 = pl.DataFrame({"source1_entity_id": ["1"], "candidate_entity_id": ["A"], "retrieval_channels": ["ngram"]})
    b3 = pl.DataFrame({"source1_entity_id": ["2"], "candidate_entity_id": ["B"], "retrieval_channels": ["exact"]})
    
    merged = union_candidates([b1, b2, b3])
    
    assert len(merged) == 2
    row1a = merged.filter((pl.col("source1_entity_id") == "1") & (pl.col("candidate_entity_id") == "A"))
    assert "exact|ngram" in row1a["retrieval_channels"][0] or "ngram|exact" in row1a["retrieval_channels"][0]
