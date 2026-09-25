import time
import os
import pickle
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.isotonic import IsotonicRegression

from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
from src.features.feature_engineering import build_features
from src.blocking.union import union_candidates
from src.blocking.exact import exact_name_blocking, exact_address_blocking
from src.blocking.structural import postal_house_blocking
from sklearn.feature_extraction.text import TfidfVectorizer

def fast_sparse_top_k(s1_lf: pl.LazyFrame, s2_s3_lf: pl.LazyFrame, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=20, channel_name="tfidf"):
    s1 = s1_lf.select(["entity_id", col_name]).collect().filter(pl.col(col_name) != "")
    s23 = s2_s3_lf.select(["entity_id", col_name]).collect().filter(pl.col(col_name) != "")
    
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=3, max_df=0.01, dtype=np.float32)
    X_s23 = vec.fit_transform(s23[col_name].to_list())
    X_s1 = vec.transform(s1[col_name].to_list())
    
    sim = X_s1.dot(X_s23.T)
    out_s1 = []
    out_s23 = []
    
    s1_ids = s1["entity_id"].to_numpy()
    s23_ids = s23["entity_id"].to_numpy()
    
    for row_idx in range(sim.shape[0]):
        row_data = sim.data[sim.indptr[row_idx]:sim.indptr[row_idx+1]]
        row_indices = sim.indices[sim.indptr[row_idx]:sim.indptr[row_idx+1]]
        if len(row_data) > 0:
            if len(row_data) > k:
                top_k_idx = np.argpartition(row_data, -k)[-k:]
                top_k_indices = row_indices[top_k_idx]
            else:
                top_k_indices = row_indices
            out_s1.extend([s1_ids[row_idx]] * len(top_k_indices))
            out_s23.extend(s23_ids[top_k_indices])
            
    df = pl.DataFrame({
        "source1_entity_id": out_s1,
        "candidate_entity_id": out_s23,
        "retrieval_channels": [channel_name] * len(out_s1)
    })
    return df

def generate_cands(s1_ids, s1, s23):
    s1_sub = s1.filter(pl.col("entity_id").is_in(s1_ids))
    b_name = exact_name_blocking(s1_sub, s23)
    b_addr = exact_address_blocking(s1_sub, s23)
    b_ph = postal_house_blocking(s1_sub, s23)
    b_word = fast_sparse_top_k(s1_sub, s23, col_name="latin_name", analyzer="word", ngram_range=(1,1), k=20, channel_name="tfidf_name")
    return union_candidates([b_name, b_addr, b_ph, b_word])

def get_labels(cands, gt):
    gt_pairs = gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls().filter(pl.col("matched_entity_ids") != "")
    gt_pairs = gt_pairs.select(["source1_entity_id", "matched_entity_ids"]).with_columns(pl.lit(1.0).alias("label"))
    cands_df = cands.join(gt_pairs, left_on=["source1_entity_id", "candidate_entity_id"], right_on=["source1_entity_id", "matched_entity_ids"], how="left").fill_null(0.0)
    return cands_df

def compute_macro_f05(preds_df, s1_list, gt_dict):
    pred_dict = {r[0]: set(r[1]) for r in preds_df.group_by("source1_entity_id").agg(pl.col("candidate_entity_id")).iter_rows()}
    s_sum, ns_sum = 0.0, 0.0
    s_cnt, ns_cnt = 0, 0
    s_tp = 0
    
    for s1_id in s1_list:
        trues = gt_dict.get(s1_id, set())
        preds = pred_dict.get(s1_id, set())
        
        score = 0.0
        if not trues and not preds: 
            score = 1.0
            s_tp += 1
        elif not trues or not preds: score = 0.0
        else:
            tp = len(trues & preds)
            prc = tp / len(preds)
            rcl = tp / len(trues)
            score = (1.25 * prc * rcl) / (0.25 * prc + rcl) if tp > 0 else 0.0
            
        if not trues:
            s_sum += score
            s_cnt += 1
        else:
            ns_sum += score
            ns_cnt += 1
            
    return s_sum, ns_sum, s_cnt, ns_cnt, s_tp

def run_pipeline():
    print("Loading data...")
    frames = load_all_data()
    s1 = frames["train_source1"]
    s23 = pl.concat([frames["train_source2"], frames["train_source3"]], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    gt_dict = {row["source1_entity_id"]: set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set() for row in gt.iter_rows(named=True)}
    t_split, v_split = get_grouped_split(gt['source1_entity_id'].to_list())
    
    print("\n--- PART 2: Task 04 Baseline ---")
    np.random.seed(42)
    t_shuf = np.random.choice(t_split, size=len(t_split), replace=False).tolist()
    tr_ids = t_shuf[0:3000]
    val_ids = t_shuf[3000:6000]
    cal_ids = t_shuf[6000:9000]
    
    print("Generating T04 Train Cands...")
    tr_cands = get_labels(generate_cands(tr_ids, s1, s23), gt)
    val_cands = get_labels(generate_cands(val_ids, s1, s23), gt)
    
    tr_feat = build_features(tr_cands.lazy(), s1, s23)
    f_cols = [c for c in tr_feat.columns if c.startswith("f_")]
    
    # Bugfix applied: 'name' to 'business_name' in feature_engineering using wrapper
    tr_feat = tr_feat.select(f_cols).to_numpy()
    tr_lbl = tr_cands["label"].to_numpy()
    
    val_feat = build_features(val_cands.lazy(), s1, s23).select(f_cols).to_numpy()
    val_lbl = val_cands["label"].to_numpy()
    
    m_base = lgb.train({'objective': 'binary', 'verbose': -1}, lgb.Dataset(tr_feat, label=tr_lbl), num_boost_round=40)
    p_val_base = m_base.predict(val_feat)
    
    print(f"ROC-AUC: {roc_auc_score(val_lbl, p_val_base):.4f}")
    print(f"PR-AUC:  {average_precision_score(val_lbl, p_val_base):.4f}")
    
    print("\n--- PART 3: Task 04 HNM ---")
    p_tr_base = m_base.predict(tr_feat)
    fp_mask = (tr_lbl == 0)
    fp_idx = np.argsort(p_tr_base[fp_mask])[-1000:]
    w = np.ones(len(tr_lbl))
    w[np.where(fp_mask)[0][fp_idx]] = 5.0
    
    m_hnm = lgb.train({'objective': 'binary', 'verbose': -1}, lgb.Dataset(tr_feat, label=tr_lbl, weight=w), num_boost_round=40)
    p_val_hnm = m_hnm.predict(val_feat)
    print(f"HNM ROC-AUC: {roc_auc_score(val_lbl, p_val_hnm):.4f}")
    print(f"HNM PR-AUC:  {average_precision_score(val_lbl, p_val_hnm):.4f}")
    
    os.makedirs("src/modeling", exist_ok=True)
    with open("src/modeling/lgb_model.pkl", "wb") as f: pickle.dump(m_hnm, f)
        
    print("\n--- PART 4: Task 05 Calibration ---")
    cal_cands = get_labels(generate_cands(cal_ids, s1, s23), gt)
    cal_feat = build_features(cal_cands.lazy(), s1, s23).select(f_cols).to_numpy()
    p_cal_raw = m_hnm.predict(cal_feat)
    
    iso = IsotonicRegression(out_of_bounds='clip')
    p_cal_iso = iso.fit_transform(p_cal_raw, cal_cands["label"].to_numpy())
    with open("src/modeling/iso_calibrator.pkl", "wb") as f: pickle.dump(iso, f)
        
    print("\n--- PART 5/6: Holdout 20k Execution ---")
    np.random.seed(101)
    t03_tune_ids = set(np.random.choice(v_split, size=10000, replace=False).tolist())
    v_unseen = [x for x in v_split if x not in t03_tune_ids]
    np.random.seed(999)
    strict_val_ids = np.random.choice(v_unseen, size=20000, replace=False).tolist()
    
    print("Generating Holdout Cands...")
    h_cands = get_labels(generate_cands(strict_val_ids, s1, s23), gt)
    h_feat = build_features(h_cands.lazy(), s1, s23)
    p_h_raw = m_hnm.predict(h_feat.select(f_cols).to_numpy())
    p_h_iso = iso.transform(p_h_raw)
    
    preds_df = h_cands.select(["source1_entity_id", "candidate_entity_id"]).with_columns([
        pl.Series("raw_score", p_h_raw), pl.Series("iso_score", p_h_iso)
    ])
    os.makedirs("output", exist_ok=True)
    preds_df.write_csv("output/val_predictions.csv")
    
    print("\n--- PART 7: Evaluation Math ---")
    # Apply Threshold 0.55 & 1-Parent Rule
    f_preds = preds_df.filter(pl.col("iso_score") >= 0.55).sort("iso_score", descending=True).unique(subset=["candidate_entity_id"], keep="first")
    
    s_sum, ns_sum, s_cnt, ns_cnt, empty_sing = compute_macro_f05(f_preds, strict_val_ids, gt_dict)
    total_sum = s_sum + ns_sum
    tot_cnt = s_cnt + ns_cnt
    ovr = total_sum / tot_cnt
    indep = (s_cnt/tot_cnt)*(s_sum/s_cnt if s_cnt>0 else 0) + (ns_cnt/tot_cnt)*(ns_sum/ns_cnt if ns_cnt>0 else 0)
    
    print(f"Total S1: {tot_cnt}")
    print(f"Singleton Count: {s_cnt}")
    print(f"Non-Singleton Count: {ns_cnt}")
    print(f"Correctly Empty Singletons: {empty_sing}")
    print(f"Incorrectly Non-Empty Singletons: {s_cnt - empty_sing}")
    print(f"Singleton Score Sum: {s_sum:.0f}")
    print(f"Singleton F0.5: {s_sum/s_cnt:.5f}")
    print(f"Non-Singleton Score Sum: {ns_sum:.5f}")
    print(f"Non-Singleton F0.5: {ns_sum/ns_cnt:.5f}")
    print(f"Total Score Sum: {total_sum:.5f}")
    print(f"Overall Macro F0.5 (Direct): {ovr:.5f}")
    print(f"Overall Macro F0.5 (Parts):  {indep:.5f}")

if __name__ == "__main__":
    run_pipeline()
