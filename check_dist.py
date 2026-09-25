import polars as pl
from src.data.loader import load_all_data
from src.evaluation.splits import get_grouped_split
import numpy as np

frames = load_all_data()
gt = frames['train_ground_truth'].collect()
t, v = get_grouped_split(gt['source1_entity_id'].to_list())

gt = gt.with_columns(
    (pl.col("matched_entity_ids").is_null()).alias("is_sing")
)

st = gt.filter(pl.col("source1_entity_id").is_in(t))
sv = gt.filter(pl.col("source1_entity_id").is_in(v))

t_tot = len(t)
v_tot = len(v)
t_sing = st.filter(pl.col("is_sing")).height
v_sing = sv.filter(pl.col("is_sing")).height

print(f"Train Sing: {t_sing} / {t_tot} ({t_sing/t_tot:.4f})")
print(f"Val Sing: {v_sing} / {v_tot} ({v_sing/v_tot:.4f})")

# 5k calib
np.random.seed(42)
train_s1_subset = set(np.random.choice(t, size=15000, replace=False).tolist())
train_unseen = [x for x in t if x not in train_s1_subset]
np.random.seed(123)
calib_s1_ids = np.random.choice(train_unseen, size=5000, replace=False).tolist()

sc = gt.filter(pl.col("source1_entity_id").is_in(calib_s1_ids))
c_sing = sc.filter(pl.col("is_sing")).height
print(f"Calib Sing: {c_sing} / 5000 ({c_sing/5000:.4f})")

# 20k holdout
np.random.seed(101)
t03_tune_ids = set(np.random.choice(v, size=10000, replace=False).tolist())
val_unseen = [x for x in v if x not in t03_tune_ids]
np.random.seed(999)
strict_val_ids = np.random.choice(val_unseen, size=20000, replace=False).tolist()
sh = gt.filter(pl.col("source1_entity_id").is_in(strict_val_ids))
h_sing = sh.filter(pl.col("is_sing")).height
print(f"Holdout Sing: {h_sing} / 20000 ({h_sing/20000:.4f})")
