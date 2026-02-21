import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import session_info
from gears import PertData

# --- Args -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Collect ground truth for combinatorial data')
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--test_train_config_id', required=True)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--working_dir', required=True)
parser.add_argument('--result_id', required=True)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

out_dir = Path(args.working_dir) / "results" / args.result_id
# out_dir.parent.mkdir(parents=True, exist_ok=True) # No need to create out_dir.parent since it should already exist from the split step

# --- Load data --------------------------------------------------------------

pert_data_folder = Path("data/gears_pert_data")
pert_data = PertData(pert_data_folder)

known_datasets = {'norman', 'adamson', 'dixit'}
if args.dataset_name in known_datasets:
    pert_data.load(args.dataset_name)
else:
    pert_data.load(data_path=str(pert_data_folder / args.dataset_name))

adata = pert_data.adata
X = adata.X  # keep sparse until needed

# --- Per-condition stats -----------------------------------------------------

conds = adata.obs["condition"].cat.remove_unused_categories().cat.categories.tolist()
condition_col = adata.obs["condition"]

# Compute all stats in one pass per condition
means, ses, n_cells = {}, {}, {}
for cond in conds:
    mask = (condition_col == cond).values
    X_sub = X[mask]                                    # sparse slice
    arr = X_sub.toarray()                              # dense only here
    means[cond]   = arr.mean(axis=0).tolist()
    ses[cond]     = (arr.std(axis=0) / arr.shape[0]).tolist()
    n_cells[cond] = int(mask.sum())

gene_names = adata.var["gene_name"].values.tolist()

# --- Save output ------------------------------------------------------------

tmp_out_dir = Path(tempfile.mkdtemp())

files = {
    "all_predictions.json":    means,
    "all_predictions_se.json": ses,
    "n_cells.json":            n_cells,
    "gene_names.json":         gene_names,
}
for fname, data in files.items():
    (tmp_out_dir / fname).write_text(json.dumps(data, indent=4), encoding="utf-8")

shutil.move(str(tmp_out_dir), out_dir)

session_info.show()
print("Python done")