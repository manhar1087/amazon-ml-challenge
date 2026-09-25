from .scorer import compute_f05_score

def evaluate_predictions(y_true, y_pred):
    """
    Evaluates predictions and returns detailed diagnostics including overall macro F0.5,
    singleton F0.5, non-singleton F0.5, and source-specific performance.
    
    Args:
        y_true: dict, mapping S1 entity IDs to a set of true matching S2/S3 entity IDs.
        y_pred: dict, mapping S1 entity IDs to a set of predicted matching S2/S3 entity IDs.
        
    Returns:
        dict: containing the metrics.
    """
    if not y_true:
        return {}

    overall_scores = []
    singleton_scores = []
    non_singleton_scores = []
    
    for s1_id, true_matches in y_true.items():
        pred_matches = y_pred.get(s1_id, set())
        
        true_set = set(true_matches) if true_matches else set()
        pred_set = set(pred_matches) if pred_matches else set()
        
        is_true_empty = len(true_set) == 0
        is_pred_empty = len(pred_set) == 0
        
        if is_true_empty and is_pred_empty:
            score = 1.0
        elif is_true_empty and not is_pred_empty:
            score = 0.0
        elif not is_true_empty and is_pred_empty:
            score = 0.0
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            score = compute_f05_score(tp, fp, fn)
            
        overall_scores.append(score)
        
        if is_true_empty:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)
            
    metrics = {
        'macro_f05': sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
        'singleton_macro_f05': sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0,
        'non_singleton_macro_f05': sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0,
        'total_s1': len(overall_scores),
        'total_singletons': len(singleton_scores),
        'total_non_singletons': len(non_singleton_scores)
    }
    return metrics
    
def print_evaluation_report(metrics):
    """
    Prints a formatted evaluation report from the metrics dictionary.
    """
    print("="*40)
    print("       EVALUATION REPORT")
    print("="*40)
    print(f"Overall Macro F0.5:       {metrics.get('macro_f05', 0):.5f}")
    print(f"Singleton Macro F0.5:     {metrics.get('singleton_macro_f05', 0):.5f} (Count: {metrics.get('total_singletons', 0)})")
    print(f"Non-Singleton Macro F0.5: {metrics.get('non_singleton_macro_f05', 0):.5f} (Count: {metrics.get('total_non_singletons', 0)})")
    print("="*40)
