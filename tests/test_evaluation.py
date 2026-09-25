import pytest
from src.evaluation.scorer import macro_f0_5, compute_f05_score
from src.evaluation.splits import get_grouped_split
from src.evaluation.reports import evaluate_predictions

def test_exact_match():
    y_true = {'S1-1': {'S2-1', 'S3-1'}}
    y_pred = {'S1-1': {'S2-1', 'S3-1'}}
    assert macro_f0_5(y_true, y_pred) == 1.0

def test_true_singleton():
    y_true = {'S1-1': set()}
    y_pred = {'S1-1': set()}
    assert macro_f0_5(y_true, y_pred) == 1.0

def test_false_positive_singleton():
    y_true = {'S1-1': set()}
    y_pred = {'S1-1': {'S2-1'}}
    assert macro_f0_5(y_true, y_pred) == 0.0

def test_false_negative_singleton():
    y_true = {'S1-1': {'S2-1'}}
    y_pred = {'S1-1': set()}
    assert macro_f0_5(y_true, y_pred) == 0.0

def test_partial_match_precision_vs_recall():
    y_true = {'S1-1': {'A', 'B'}}
    
    # Prediction 1: 1 wrong, 0 missing (High recall, lower precision)
    # TP=2, FP=1, FN=0 -> P=2/3, R=1.0 -> F0.5 = 1.25*(2/3)*1 / (0.25*(2/3) + 1) = 0.833 / 1.166 = 0.714
    y_pred_1 = {'S1-1': {'A', 'B', 'C'}}
    
    # Prediction 2: 0 wrong, 1 missing (High precision, lower recall)
    # TP=1, FP=0, FN=1 -> P=1.0, R=0.5 -> F0.5 = 1.25*(1)*0.5 / (0.25*1 + 0.5) = 0.625 / 0.75 = 0.833
    y_pred_2 = {'S1-1': {'A'}}
    
    score1 = macro_f0_5(y_true, y_pred_1)
    score2 = macro_f0_5(y_true, y_pred_2)
    
    # F0.5 favors precision, so Prediction 2 should have a higher score
    assert score2 > score1
    assert abs(score1 - 0.714) < 0.01
    assert abs(score2 - 0.833) < 0.01

def test_macro_averaging():
    y_true = {
        'S1-1': {'A', 'B'}, # will score 1.0
        'S1-2': set(),      # will score 1.0
        'S1-3': set(),      # will score 0.0 (FP)
    }
    y_pred = {
        'S1-1': {'A', 'B'},
        'S1-2': set(),
        'S1-3': {'C'},
    }
    
    score = macro_f0_5(y_true, y_pred)
    assert abs(score - 0.666) < 0.01

def test_splits_determinism():
    ids = [f'S1-{i}' for i in range(1000)]
    train1, val1 = get_grouped_split(ids, test_size=0.2, seed=42)
    train2, val2 = get_grouped_split(ids, test_size=0.2, seed=42)
    
    assert train1 == train2
    assert val1 == val2
    assert abs(len(val1) - 200) < 30 # roughly 20%

def test_reports():
    y_true = {
        'S1-1': {'A', 'B'},
        'S1-2': set(),
        'S1-3': set(),
    }
    y_pred = {
        'S1-1': {'A', 'B'},
        'S1-2': set(),
        'S1-3': {'C'},
    }
    metrics = evaluate_predictions(y_true, y_pred)
    assert metrics['total_s1'] == 3
    assert metrics['total_singletons'] == 2
    assert metrics['total_non_singletons'] == 1
    assert abs(metrics['non_singleton_macro_f05'] - 1.0) < 1e-6
    assert abs(metrics['singleton_macro_f05'] - 0.5) < 1e-6
    assert abs(metrics['macro_f05'] - (1.0 + 1.0 + 0.0) / 3) < 1e-6

def test_one_false_positive():
    y_true = {'S1-1': {'A'}}
    y_pred = {'S1-1': {'A', 'B'}}
    # TP=1, FP=1, FN=0 -> Precision=0.5, Recall=1.0
    # F0.5 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555
    score = macro_f0_5(y_true, y_pred)
    assert abs(score - 0.5555) < 0.01

def test_one_false_negative():
    y_true = {'S1-1': {'A', 'B'}}
    y_pred = {'S1-1': {'A'}}
    # TP=1, FP=0, FN=1 -> Precision=1.0, Recall=0.5
    # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333
    score = macro_f0_5(y_true, y_pred)
    assert abs(score - 0.8333) < 0.01

def test_multiple_tp_and_fp():
    y_true = {'S1-1': {'A', 'B', 'C', 'D'}}
    y_pred = {'S1-1': {'A', 'B', 'E', 'F'}}
    # TP=2, FP=2, FN=2 -> Precision=0.5, Recall=0.5
    # F0.5 = 1.25 * 0.5 * 0.5 / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    score = macro_f0_5(y_true, y_pred)
    assert abs(score - 0.5) < 0.01

def test_duplicate_predicted_ids():
    y_true = {'S1-1': {'A', 'B'}}
    # Predictions provided as a list with duplicates
    y_pred = {'S1-1': ['A', 'A', 'B', 'B', 'B']}
    # Scorer should treat this identically to {'A', 'B'}
    score = macro_f0_5(y_true, y_pred)
    assert score == 1.0
