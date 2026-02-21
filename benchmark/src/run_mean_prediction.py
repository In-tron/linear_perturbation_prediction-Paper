import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import anndata as ad
import session_info

# --- Args -------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run model that always predicts the mean across conditions")
parser.add_argument("--dataset_name", required=True)
parser.add_argument("--test_train_config_id", required=True)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--working_dir", required=True)
parser.add_argument("--result_id", required=True)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

out_dir = Path(args.working_dir) / "results" / args.result_id
# out_dir.parent.mkdir(parents=True, exist_ok=True) # No need to create out_dir.parent since it should already exist from the split step

# --- Load data --------------------------------------------------------------

adata = ad.read_h5ad(f"data/gears_pert_data/{args.dataset_name}/perturb_processed.h5ad")

with open(Path(args.working_dir) / "results" / args.test_train_config_id) as f:
    set2condition = json.load(f)

if "ctrl" not in set2condition["train"]:
    set2condition["train"].append("ctrl")

all_conditions = {c for conds in set2condition.values() for c in conds}
adata = adata[adata.obs["condition"].isin(all_conditions)].copy()
adata.obs["condition"] = adata.obs["condition"].astype("category").cat.remove_unused_categories()
adata.obs["clean_condition"] = adata.obs["condition"].str.replace(r"\+ctrl$", "", regex=True)

training_map = {cond: split for split, conds in set2condition.items() for cond in conds}
adata.obs["training"] = adata.obs["condition"].map(training_map)

gene_names = adata.var["gene_name"].values.tolist()
adata.var_names = gene_names

# --- Pseudobulk -------------------------------------------------------------

X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()  # (cells, genes)

groups = (adata.obs[["condition", "clean_condition", "training"]]
          .drop_duplicates().reset_index(drop=True))

pb_X = np.zeros((len(groups), X.shape[1]))  # (groups, genes)
for i, row in groups.iterrows():
    mask = (adata.obs["condition"] == row["condition"]).values
    pb_X[i] = X[mask].mean(axis=0)

# --- Mean across training conditions ----------------------------------------

train_mask = (groups["training"] == "train").values
mean_across_training = pb_X[train_mask].mean(axis=0)  # (genes,)

# Assign that same vector to every condition
pred = {cond: mean_across_training.tolist()
        for cond in groups["clean_condition"]}

# --- Save output ------------------------------------------------------------

tmp_out_dir = Path(tempfile.mkdtemp())
(tmp_out_dir / "all_predictions.json").write_text(json.dumps(pred), encoding="utf-8")
(tmp_out_dir / "gene_names.json").write_text(json.dumps(gene_names), encoding="utf-8")

if out_dir.exists():
    shutil.rmtree(out_dir)
shutil.move(str(tmp_out_dir), out_dir)

session_info.show()
print("Python done")