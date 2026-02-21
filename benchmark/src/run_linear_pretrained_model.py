import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import anndata as ad
import session_info

# --- Args -------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run linear pretrained model")
parser.add_argument("--dataset_name", required=True, type=str)
parser.add_argument("--test_train_config_id", required=True, type=str)
parser.add_argument("--pca_dim", default=10, type=int)
parser.add_argument("--ridge_penalty", default=0.1, type=float)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--gene_embedding", default="training_data", type=str)
parser.add_argument("--pert_embedding", default="training_data", type=str)
parser.add_argument("--working_dir", required=True, type=str)
parser.add_argument("--result_id", required=True, type=str)
args = parser.parse_args()
print(args)

np.random.seed(args.seed)

out_dir = Path(args.working_dir) / "results" / args.result_id
out_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path(args.working_dir) / "results" / args.test_train_config_id
# data_dir.mkdir(parents=True, exist_ok=True) # No need to create data_dir since it should already exist from the split step


# --- Core solver ------------------------------------------------------------

def solve_y_axb(Y, A=None, B=None, A_ridge=0.01, B_ridge=0.01):
    """Solve Y ≈ A @ K @ B^T via ridge regression.
    
    Y: (genes, perts)
    A: (genes, gene_dims)
    B: (perts, pert_dims)
    Returns K: (gene_dims, pert_dims)
    """
    assert isinstance(Y, np.ndarray), "Y must be a numpy array"
    center = Y.mean(axis=1)        # (genes,)
    Y = Y - center[:, None]

    def ridge_solve(M, lam):
        """(M^T M + lam I)^{-1} M^T  →  (dims, samples)"""
        return np.linalg.solve(M.T @ M + lam * np.eye(M.shape[1]), M.T)

    if A is not None and B is not None:
        # K = (A^T A + lam I)^{-1} A^T Y B (B^T B + lam I)^{-1}
        K = ridge_solve(A, A_ridge) @ Y @ B @ np.linalg.solve(
            B.T @ B + B_ridge * np.eye(B.shape[1]), np.eye(B.shape[1])
        )
    elif B is None:
        K = ridge_solve(A, A_ridge) @ Y
    elif A is None:
        K = Y @ B @ np.linalg.solve(
            B.T @ B + B_ridge * np.eye(B.shape[1]), np.eye(B.shape[1])
        )
    else:
        raise ValueError("At least one of A or B must be non-None")

    K = np.nan_to_num(K)
    return K, center


# --- Load data --------------------------------------------------------------

folder = Path("data/gears_pert_data")
adata = ad.read_h5ad(folder / args.dataset_name / "perturb_processed.h5ad")

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

gene_names = adata.var["gene_name"].values
adata.var_names = gene_names

# AnnData X is (cells, genes)
X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()  # (cells, genes)
baseline = X[adata.obs["condition"] == "ctrl", :].mean(axis=0)          # (genes,)


# --- Pseudobulk -------------------------------------------------------------

def pseudobulk(adata, X, group_cols):
    """Average expression per group. Returns pb_X: (groups, genes)."""
    groups = adata.obs[group_cols].drop_duplicates().reset_index(drop=True)
    pb_X = np.zeros((len(groups), X.shape[1]))
    for i, row in groups.iterrows():
        mask = np.ones(adata.n_obs, dtype=bool)
        for col in group_cols:
            mask &= (adata.obs[col] == row[col]).values
        pb_X[i, :] = X[mask].mean(axis=0)
    return pb_X, groups.reset_index(drop=True)


pb_X, pb_obs = pseudobulk(adata, X, ["condition", "clean_condition", "training"])
pb_change = pb_X - baseline[None, :]   # (groups, genes)

train_mask = pb_obs["training"] == "train"
train_X      = pb_X[train_mask.values, :]       # (train_perts, genes)
train_change = pb_change[train_mask.values, :]
train_obs    = pb_obs[train_mask].reset_index(drop=True)


# --- Embeddings -------------------------------------------------------------
# Convention (matching solver):
#   gene_emb : (genes,   gene_dims)   — A
#   pert_emb : (perts,   pert_dims)   — B

def make_gene_embedding(mode, train_X, gene_names, pca_dim):
    """train_X: (perts, genes)  →  embedding: (genes, dims)"""
    n_genes = train_X.shape[1]
    if mode == "training_data":
        svd = TruncatedSVD(n_components=pca_dim, random_state=0)
        emb = svd.fit_transform(train_X.T)   # (genes, pca_dim)
        return pd.DataFrame(emb, index=gene_names)
    elif mode == "identity":
        return pd.DataFrame(np.eye(n_genes), index=gene_names)
    elif mode == "zero":
        return pd.DataFrame(np.zeros((n_genes, n_genes)), index=gene_names)
    elif mode == "random":
        return pd.DataFrame(np.random.randn(n_genes, pca_dim), index=gene_names)
    else:
        df = pd.read_csv(mode, sep="\t", index_col=0)
        return df   # expected (genes, dims)


def make_pert_embedding(mode, train_X, clean_conditions, pca_dim):
    """train_X: (perts, genes)  →  embedding: (perts, dims)"""
    n_perts = train_X.shape[0]
    if mode == "training_data":
        svd = TruncatedSVD(n_components=pca_dim, random_state=0)
        emb = svd.fit_transform(train_X)     # (perts, pca_dim)
        return pd.DataFrame(emb, index=clean_conditions)
    elif mode == "identity":
        return pd.DataFrame(np.eye(n_perts), index=clean_conditions)
    elif mode == "zero":
        return pd.DataFrame(np.zeros((n_perts, n_perts)), index=clean_conditions)
    elif mode == "random":
        return pd.DataFrame(np.random.randn(n_perts, pca_dim), index=clean_conditions)
    else:
        df = pd.read_csv(mode, sep="\t", index_col=0)
        return df   # expected (perts, dims)


gene_emb = make_gene_embedding(args.gene_embedding, train_X, gene_names, args.pca_dim)
pert_emb = make_pert_embedding(args.pert_embedding, train_X, train_obs["clean_condition"].values, args.pca_dim)

if "ctrl" not in pert_emb.index:
    pert_emb.loc["ctrl"] = 0.0

# Match embeddings to training data
gene_match_mask = gene_emb.index.isin(gene_names)
pert_match_mask = pert_emb.index.isin(train_obs["clean_condition"])

if pert_match_mask.sum() <= 1:
    raise ValueError("Too few matches between clean_conditions and pert_embedding")
if gene_match_mask.sum() <= 1:
    raise ValueError("Too few matches between gene names and gene_embedding")

gene_emb_sub = gene_emb.loc[gene_match_mask].values   # (matched_genes, gene_dims)
pert_emb_sub = pert_emb.loc[pert_match_mask].values   # (matched_perts, pert_dims)

matched_genes = gene_emb.index[gene_match_mask]
matched_perts = pert_emb.index[pert_match_mask]

gene_idx = [list(gene_names).index(g) for g in matched_genes]
pert_idx  = [list(train_obs["clean_condition"]).index(p) for p in matched_perts]

# Y: (matched_genes, matched_perts)
Y = train_change[np.ix_(pert_idx, gene_idx)].T


# --- Fit model --------------------------------------------------------------

K, center = solve_y_axb(Y, A=gene_emb_sub, B=pert_emb_sub,
                         A_ridge=args.ridge_penalty, B_ridge=args.ridge_penalty)

# Predict for all conditions: pred = A K B^T + center + baseline
# gene_emb_sub: (matched_genes, gene_dims)
# K:            (gene_dims, pert_dims)
# pert_emb_all: (all_perts, pert_dims)
all_clean_conds = pb_obs["clean_condition"].values
pert_emb_all = pert_emb.reindex(all_clean_conds, fill_value=0.0).values  # (all_perts, pert_dims)

baseline_sub = baseline[gene_idx]                                          # (matched_genes,)
# pred: (matched_genes, all_perts)
pred = gene_emb_sub @ K @ pert_emb_all.T + center[:, None] + baseline_sub[:, None]


# --- Summary statistics -----------------------------------------------------

# obs_change / pred_change both: (matched_genes, all_perts)
obs_change  = pb_change[np.ix_(range(len(pb_obs)), gene_idx)].T  # (matched_genes, all_perts)
pred_change = pred - baseline_sub[:, None]

r2_rows = []
for i, (cond, split) in enumerate(zip(all_clean_conds, pb_obs["training"])):
    obs_i, pred_i = obs_change[:, i], pred_change[:, i]
    valid = ~(np.isnan(obs_i) | np.isnan(pred_i))
    if valid.sum() > 1:
        r2 = np.corrcoef(obs_i[valid], pred_i[valid])[0, 1]
        r2_rows.append({"cond": cond, "training": split, "r2": r2})

r2_df = pd.DataFrame(r2_rows)
print(r2_df.groupby("training")["r2"].describe())


# --- Save output ------------------------------------------------------------

tmp_out_dir = Path(tempfile.mkdtemp()) / "prediction_storage"
tmp_out_dir.mkdir(parents=True)

pred_dict = {cond: pred[:, i].tolist() for i, cond in enumerate(all_clean_conds)}
(tmp_out_dir / "all_predictions.json").write_text(json.dumps(pred_dict))
(tmp_out_dir / "gene_names.json").write_text(json.dumps(list(matched_genes)))

if out_dir.exists():
    shutil.rmtree(out_dir)
shutil.move(str(tmp_out_dir), str(out_dir))

session_info.show()
print("Python done")