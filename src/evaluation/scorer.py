def compute_f05_score(tp, fp, fn):
    """
    Computes the F0.5 score given True Positives, False Positives, and False Negatives.
    """
    if tp == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    # F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def macro_f0_5(y_true, y_pred):
    """
    Calculates the macro F0.5 score per S1 entity.
    
    Args:
        y_true: dict, mapping S1 entity IDs to a set/list of true matching S2/S3 entity IDs.
        y_pred: dict, mapping S1 entity IDs to a set/list of predicted matching S2/S3 entity IDs.
        
    Returns:
        float: the macro-averaged F0.5 score across all S1 entities in y_true.
    """
    if not y_true:
        return 0.0

    scores = []
    
    for s1_id, true_matches in y_true.items():
        # Get predictions, default to empty set if not present
        pred_matches = y_pred.get(s1_id, set())
        
        # Ensure they are sets for set operations
        true_set = set(true_matches) if true_matches else set()
        pred_set = set(pred_matches) if pred_matches else set()
        
        is_true_empty = len(true_set) == 0
        is_pred_empty = len(pred_set) == 0
        
        # Edge cases explicitly defined
        if is_true_empty and is_pred_empty:
            scores.append(1.0)
            continue
        if is_true_empty and not is_pred_empty:
            scores.append(0.0)
            continue
        if not is_true_empty and is_pred_empty:
            scores.append(0.0)
            continue
            
        # Standard case
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        scores.append(compute_f05_score(tp, fp, fn))
        
    return sum(scores) / len(scores)
