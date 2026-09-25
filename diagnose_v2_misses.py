import polars as pl
from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split

def run_v2_diagnosis():
    print("Loading data...")
    frames = load_all_data()
    
    s1 = frames["train_source1"].collect()
    s23 = pl.concat([frames["train_source2"].collect(), frames["train_source3"].collect()], how="diagonal")
    gt = frames["train_ground_truth"].collect()
    
    s1_ids_all = gt["source1_entity_id"].to_list()
    train_s1_ids, val_s1_ids = get_grouped_split(s1_ids_all)
    val_s1_set = set(val_s1_ids)
    
    try:
        cands_df = pl.read_csv("output/candidate_pairs_v2.tsv", separator="\t")
    except Exception as e:
        print(f"Could not load candidates: {e}")
        return
        
    cands_dict = {}
    for row in cands_df.iter_rows(named=True):
        cands = row["candidate_entity_ids"]
        cands_dict[row["source1_entity_id"]] = set(cands.split(",")) if cands else set()
        
    gt_exploded = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .filter(pl.col("source1_entity_id").is_in(list(val_s1_set)))
    )
    
    gt_pairs = gt_exploded.select(["source1_entity_id", "matched_entity_ids"]).to_dicts()
    
    missed_pairs = []
    for row in gt_pairs:
        s1_id = row["source1_entity_id"]
        s23_id = row["matched_entity_ids"]
        if s23_id not in cands_dict.get(s1_id, set()):
            missed_pairs.append((s1_id, s23_id))
            
    print(f"Total Missed: {len(missed_pairs)}")
    if len(missed_pairs) == 0:
        return
        
    missed_df = pl.DataFrame(missed_pairs, schema=["source1_entity_id", "matched_entity_ids"])
    mf = missed_df.join(s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
    mf = mf.join(s23, left_on="matched_entity_ids", right_on="entity_id", how="inner", suffix="_right")
    
    print("\nSample Misses (Name Variation / Abbreviation):")
    samples = mf.head(10).to_dicts()
    for row in samples:
        print(f"S1 Name: {row['name']} | Addr: {row['business_address']}")
        print(f"S23 Name: {row['name_right']} | Addr: {row['business_address_right']}\n")

if __name__ == "__main__":
    run_v2_diagnosis()
