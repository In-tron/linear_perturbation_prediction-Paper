import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scanpy as sc
import session_info

from gears import PertData
import gears.version
assert gears.version.__version__ == '0.1.2'


# --- Args -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Prepare data for combinatorial perturbation prediction')
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--working_dir', required=True)
parser.add_argument('--result_id', required=True)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

PERT_DATA_FOLDER = Path("data/gears_pert_data")
outfile = Path(args.working_dir) / "results" / args.result_id



# --- Helpers ----------------------------------------------------------------

def normalize_condition_names(obs):
    import pandas as pd
    obs['condition'] = pd.Categorical(
        ["+".join(sorted(x.split("+"))) for x in obs['condition']],
        categories=sorted({"+".join(sorted(x.split("+"))) for x in obs['condition']})
    )
    return obs


def split_double_perts(double_pert, seed):
    """Split double perturbations into train/test/val halves."""
    train = np.random.choice(double_pert, size=len(double_pert) // 2, replace=False).tolist()
    remaining = np.setdiff1d(double_pert, train).tolist()
    test = remaining[:len(remaining) // 2]
    val = remaining[len(remaining) // 2:]
    return train, test, val


def load_pert_data(dataset_name, data_path=None):
    pd = PertData(PERT_DATA_FOLDER)
    pd.load(dataset_name if data_path is None else data_path)
    return pd


# --- Dataset-specific setup -------------------------------------------------

def setup_norman(pert_data):
    adata = pert_data.adata
    new_obs = normalize_condition_names(adata.obs.copy())
    if not adata.obs.equals(new_obs):
        adata.obs = new_obs
        adata.write_h5ad(PERT_DATA_FOLDER / "norman/perturb_processed.h5ad")
        pyg_folder = PERT_DATA_FOLDER / "norman" / "data_pyg"
        if pyg_folder.exists():
            (pyg_folder / "cell_graphs.pkl").unlink(missing_ok=True)
            pyg_folder.rmdir()
        pert_data = load_pert_data("norman")
    return pert_data


def download_norman_scfoundation():
    dest = PERT_DATA_FOLDER / "norman_from_scfoundation" / "perturb_processed.h5ad"
    if dest.exists():
        return
    url = "https://figshare.com/ndownloader/files/44477939"
    zip_path = Path("data/norman_from_scfoundation_data.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("data/norman_from_scfoundation_data")
    gene2go_dst = PERT_DATA_FOLDER / "gene2go.pkl"
    if not gene2go_dst.exists():
        shutil.copyfile(
            "/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark"
            "/tmp/scfoundation/scfoundation_gears/data/gene2go.pkl",
            gene2go_dst
        )
    adata = sc.read_h5ad(
        "data/norman_from_scfoundation_data/scFoundation/GEARS/data"
        "/gse133344_k562gi_oe_pert227_84986_19264_withtotalcount.h5ad"
    )
    adata.uns.setdefault('log1p', {})['base'] = None
    pd = PertData(PERT_DATA_FOLDER)
    pd.new_data_process(dataset_name="norman_from_scfoundation", adata=adata)


# --- Main -------------------------------------------------------------------

if args.dataset_name == "norman_from_scfoundation":
    import sys
    sys.path.insert(0, "/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/scfoundation_gears/")
    sys.path.append("/g/huber/users/ahlmanne/projects/perturbation_prediction-benchmark/tmp/scfoundation/model/")
    download_norman_scfoundation()

pert_data = load_pert_data(
    args.dataset_name,
    data_path=f"data/gears_pert_data/{args.dataset_name}" if args.dataset_name == "norman_from_scfoundation" else None
)

if args.dataset_name == "norman":
    pert_data = setup_norman(pert_data)

if args.dataset_name in ("norman", "norman_from_scfoundation"):
    conds = pert_data.adata.obs['condition'].cat.remove_unused_categories().cat.categories.tolist()
    single_pert = [c for c in conds if 'ctrl' in c]
    double_pert = np.setdiff1d(conds, single_pert).tolist()
    train, test, val = split_double_perts(double_pert, args.seed)
    set2conditions = {"train": single_pert + train, "test": test, "val": val}
else:
    pert_data.prepare_split(split='simulation', seed=args.seed)
    set2conditions = pert_data.set2conditions

outfile.parent.mkdir(parents=True, exist_ok=True)
with open(outfile, "w") as f:
    json.dump(set2conditions, f)

session_info.show()
print("Python done")