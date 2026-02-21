import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import session_info

from gears import PertData, GEARS

# --- Args -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Run GEARS')
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--test_train_config_id', required=True)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--working_dir', required=True)
parser.add_argument('--result_id', required=True)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

out_dir = Path(args.working_dir) / "results" / args.result_id
out_dir.parent.mkdir(parents=True, exist_ok=True)

# --- Version-gated imports --------------------------------------------------

# if args.dataset_name == "norman_from_scfoundation":
#     import sys
#     sys.path.insert(0, "/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/scfoundation_gears/")
#     sys.path.append("/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/model/")
#     expected_version = '0.0.2'
# else:
#     expected_version = '0.1.2'

# import gears.version
# assert gears.version.__version__ == expected_version, (
#     f"Expected GEARS {expected_version}, got {gears.version.__version__}"
# )

# --- Load data --------------------------------------------------------------

pert_data_folder = Path("data/gears_pert_data")
pert_data = PertData(pert_data_folder)

known_datasets = {'norman', 'adamson', 'dixit'}
if args.dataset_name in known_datasets:
    pert_data.load(args.dataset_name)
else:
    pert_data.load(data_path=str(pert_data_folder / args.dataset_name))

with open(Path(args.working_dir) / "results" / args.test_train_config_id) as f:
    set2conditions = json.load(f)
print(set2conditions)

pert_data.set2conditions = set2conditions
pert_data.split = "custom"
pert_data.subgroup = None
pert_data.seed = args.seed
pert_data.train_gene_set_size = 0.75
pert_data.get_dataloader(batch_size=32, test_batch_size=128)

# --- Train ------------------------------------------------------------------

gears_model = GEARS(pert_data, device='cuda')
gears_model.model_initialize(hidden_size=64)
gears_model.train(epochs=args.epochs)

# --- Predict ----------------------------------------------------------------

conds = pert_data.adata.obs["condition"].cat.remove_unused_categories().cat.categories.tolist()
split_conds = [
    [g for g in cond.split("+") if g != "ctrl"]
    for cond in conds
]

all_pred_vals = {k: v.tolist() for k, v in gears_model.predict(split_conds).items()}
gene_names = pert_data.adata.var["gene_name"].values.tolist()

# --- Save output ------------------------------------------------------------

tmp_out_dir = Path(tempfile.mkdtemp())
gears_model.save_model(str(tmp_out_dir / "gears_model"))
(tmp_out_dir / "all_predictions.json").write_text(json.dumps(all_pred_vals, indent=4), encoding="utf-8")
(tmp_out_dir / "gene_names.json").write_text(json.dumps(gene_names, indent=4), encoding="utf-8")

if out_dir.exists():
    shutil.rmtree(out_dir)
shutil.move(str(tmp_out_dir), out_dir)

session_info.show()
print("Python done")