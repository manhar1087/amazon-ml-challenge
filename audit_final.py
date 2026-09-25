import polars as pl
import numpy as np
from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score

def audit():
    print("Loading GT and Predictions...")
    preds = pl.read_csv("output/val_predictions.csv")
    frames = load_all_data()
    gt = frames["train_ground_truth"].collect()
    
    t_split, v_split = get_grouped_split(gt['source1_entity_id'].to_list())
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(v_split, size=10000, replace=False).tolist())
    v_unseen = [x for x in v_split if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(v_unseen, size=20000, replace=False).tolist()
    
    # 1. Overlap Audit (by construction, just print)
    print("\n--- PART 1: Overlap Audit ---")
    tr_ids = set(t_split[0:3000])
    val_ids = set(t_split[3000:6000])
    cal_ids = set(t_split[6000:9000])
    h_ids = set(strict_val_ids)
    
    print(f"T03 Tune INTERSECT Holdout: {len(t03_tune_ids & h_ids)}")
    print(f"T04 Train INTERSECT Holdout: {len(tr_ids & h_ids)}")
    print(f"T04 Val INTERSECT Holdout: {len(val_ids & h_ids)}")
    print(f"T05 Calib INTERSECT Holdout: {len(cal_ids & h_ids)}")
    print(f"HNM INTERSECT Holdout: {len(tr_ids & h_ids)}")
    
    # 2. Candidate Recall
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for row in gt.iter_rows(named=True)}
    
    cands_dict = {r[0]: set(r[1]) for r in preds.group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
    
    tot_true = 0
    tot_found = 0
    tot_missed = 0
    s1_cand_counts = []
    
    for s1 in strict_val_ids:
        t = gt_dict.get(s1, set())
        c = cands_dict.get(s1, set())
        s1_cand_counts.append(len(c))
        
        tot_true += len(t)
        found = len(t & c)
        tot_found += found
        tot_missed += (len(t) - found)
        
    print("\n--- PART 2: Candidate Recall (20k Holdout) ---")
    print(f"Total true matched IDs: {tot_true}")
    print(f"True IDs present in candidates: {tot_found}")
    print(f"True IDs missed by blocking: {tot_missed}")
    print(f"Pair candidate recall: {tot_found/tot_true if tot_true>0 else 0:.4f}")
    
    # 3. Attribution
    gt_exploded = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "")
    gt_pairs = gt_exploded.select(["source1_entity_id", "matched_entity_ids"]).with_columns(pl.lit(1.0).alias("label"))
    preds_lbl = preds.join(gt_pairs, left_on=["source1_entity_id", "candidate_entity_id"], right_on=["source1_entity_id", "matched_entity_ids"], how="left").fill_null(0.0)
    
    y = preds_lbl["label"].to_numpy()
    p_iso = preds_lbl["iso_score"].to_numpy()
    
    # Precision/Recall at 0.55
    y_pred = (p_iso >= 0.55).astype(int)
    roc = roc_auc_score(y, p_iso)
    prc = average_precision_score(y, p_iso)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    
    print("\n--- PART 3: Pairwise Classifier ---")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {prc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    
    # Final Decision Policy Stats
    def eval_policy(preds_df):
        p_dict = {r[0]: set(r[1]) for r in preds_df.group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
        s_sum, ns_sum = 0.0, 0.0
        fp_tot, fn_tot, p_tot = 0, 0, 0
        s_tp = 0
        s_cnt = 0
        f_scores = []
        for s1 in strict_val_ids:
            t = gt_dict.get(s1, set())
            p = p_dict.get(s1, set())
            
            fp_tot += len(p - t)
            fn_tot += len(t - p)
            p_tot += len(p)
            
            score = 0.0
            if not t and not p:
                score = 1.0
                s_tp += 1
            elif not t or not p: score = 0.0
            else:
                tp = len(t & p)
                pr = tp / len(p)
                rc = tp / len(t)
                score = (1.25 * pr * rc) / (0.25 * pr + rc) if tp > 0 else 0.0
                
            f_scores.append(score)
            if not t: s_cnt += 1
            
        return p_tot, fp_tot/20000, fn_tot/20000, s_tp/s_cnt, np.mean(f_scores)
        
    f_iso_1p = preds.filter(pl.col("iso_score") >= 0.55).sort("iso_score", descending=True).unique(subset=["candidate_entity_id"], keep="first")
    p_tot, fp_avg, fn_avg, sing_acc, mac = eval_policy(f_iso_1p)
    print("\n--- PART 3C: Final Decision Policy ---")
    print(f"Predicted pairs: {p_tot}")
    print(f"Avg FP/S1: {fp_avg:.4f}")
    print(f"Avg FN/S1: {fn_avg:.4f}")
    print(f"Singleton Empty Acc: {sing_acc:.4f}")
    print(f"Final Macro F0.5: {mac:.5f}")
    
    print("\n--- PART 4: Compare Stages ---")
    _, _, _, _, m1 = eval_policy(preds.filter(pl.col("raw_score") >= 0.5))
    _, _, _, _, m2 = eval_policy(preds.filter(pl.col("iso_score") >= 0.55))
    print(f"Raw >= 0.5: {m1:.5f}")
    print(f"Iso >= 0.55: {m2:.5f}")
    print(f"Iso >= 0.55 + 1Parent: {mac:.5f}")

if __name__ == "__main__":
    audit()
