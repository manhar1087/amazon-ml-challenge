import time
import os
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.features.feature_engineering import build_features
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.union import union_candidates

def run_fix():
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    t, v = get_grouped_split(gt['source1_entity_id'].to_list())
    
    gt_exploded = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls()
    gt_pairs = gt_exploded.select(["source1_entity_id", "matched_entity_ids"]).with_columns(pl.lit(1.0).alias("label"))
    
    # Fast train set
    np.random.seed(42)
    train_s1 = np.random.choice(t, size=2000, replace=False).tolist()
    s1_tr = s1.filter(pl.col("entity_id").is_in(train_s1))
    c_name = exact_name_blocking(s1_tr, s23)
    c_addr = exact_address_blocking(s1_tr, s23)
    c_tr = union_candidates([c_name, c_addr])
    
    c_tr = c_tr.join(gt_pairs, left_on=["source1_entity_id", "candidate_entity_id"], right_on=["source1_entity_id", "matched_entity_ids"], how="left").fill_null(0.0)
    
    # Train
    f_tr = build_features(c_tr.lazy(), s1, s23)
    f_cols = [c for c in f_tr.columns if c.startswith("f_")]
    m = lgb.train({'objective': 'binary', 'verbose': -1}, lgb.Dataset(f_tr.select(f_cols).to_numpy(), label=f_tr["label"].to_numpy()), num_boost_round=10)
    
    # Holdout 20k
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(v, size=10000, replace=False).tolist())
    val_unseen = [x for x in v if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(val_unseen, size=20000, replace=False).tolist()
    
    s1_val = s1.filter(pl.col("entity_id").is_in(strict_val_ids))
    v_name = exact_name_blocking(s1_val, s23)
    v_addr = exact_address_blocking(s1_val, s23)
    c_val = union_candidates([v_name, v_addr])
    
    f_val = build_features(c_val.lazy(), s1, s23)
    p_raw = m.predict(f_val.select(f_cols).to_numpy())
    
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(f_tr["label"].to_numpy(), f_tr["label"].to_numpy()) # Dummy fit
    p_iso = p_raw # Just use raw for speed
    
    preds = f_val.select(["source1_entity_id", "candidate_entity_id"]).with_columns(pl.Series("iso_score", p_iso))
    preds = preds.filter(pl.col("iso_score") >= 0.55).sort("iso_score", descending=True).unique(subset=["candidate_entity_id"], keep="first")
    
    pred_dict = {r[0]: set(r[1]) for r in preds.group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for row in gt.iter_rows(named=True)}
    
    s_sum = 0.0
    ns_sum = 0.0
    s_cnt = 0
    ns_cnt = 0
    
    for s1_id in strict_val_ids:
        trues = gt_dict.get(s1_id, set())
        p = pred_dict.get(s1_id, set())
        
        score = 0.0
        if not trues and not p: score = 1.0
        elif not trues or not p: score = 0.0
        else:
            tp = len(trues & p)
            prc = tp / len(p)
            rcl = tp / len(trues)
            score = (1.25 * prc * rcl) / (0.25 * prc + rcl) if tp > 0 else 0.0
            
        if not trues:
            s_sum += score
            s_cnt += 1
        else:
            ns_sum += score
            ns_cnt += 1
            
    print(f"Singleton Count: {s_cnt}")
    print(f"Singleton Score Sum: {s_sum:.0f}")
    print(f"Non-Singleton Count: {ns_cnt}")
    print(f"Non-Singleton Score Sum: {ns_sum:.3f}")
    print(f"Total Sum: {s_sum + ns_sum:.3f}")
    print(f"Overall F0.5: {(s_sum + ns_sum)/20000:.4f}")

if __name__ == "__main__":
    run_fix()
