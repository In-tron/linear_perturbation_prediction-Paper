import argparse
import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# 1. Argument Parsing
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run linear pretrained model")
parser.add_argument("--dataset_name", type=str, help="The name of the dataset")
parser.add_argument("--test_train_config_id", type=str, help="The ID of the test/train/holdout run")
parser.add_argument("--pca_dim", type=int, default=10, help="The number of PCA dimensions")
parser.add_argument("--ridge_penalty", type=float, default=0.1, help="The ridge penalty")
parser.add_argument("--seed", type=int, default=1, help="The seed")
parser.add_argument("--gene_embedding", type=str, default="training_data", 
                    help="Path to tsv with gene embedding or 'training_data'")
parser.add_argument("--pert_embedding", type=str, default="training_data", 
                    help="Path to tsv with perturbation embedding or 'training_data'")
parser.add_argument("--working_dir", type=str, help="Directory containing params, results, scripts etc.")
parser.add_argument("--result_id", type=str, help="The result_id")

args = parser.parse_args()

# Print arguments to verify
print(args)

# Set seed
np.random.seed(args.seed)

# Define Output Directory
out_dir = os.path.join(args.working_dir, "results", args.result_id)

# ------------------------------------------------------------------------------
# 2. Helper Functions
# ------------------------------------------------------------------------------

def solve_y_axb(Y, A=None, B=None, A_ridge=0.01, B_ridge=0.01):
    """
    Solves for K in Y = A * K * B.T using Ridge regression logic.
    Y: Genes x Conditions (Target change)
    A: Genes x Dim_A (Gene Embedding)
    B: Conditions x Dim_B (Perturbation Embedding)
    """
    # Ensure inputs are numpy arrays
    Y = np.array(Y)
    if A is not None: A = np.array(A)
    if B is not None: B = np.array(B)

    # Center Y (Gene-wise centering)
    center = Y.mean(axis=1, keepdims=True)
    Y_centered = Y - center

    if A is not None and B is not None:
        # K = (A'A + lambda*I)^-1 * A' * Y * B * (B'B + lambda*I)^-1
        # Solve LHS: (A'A + Id) * X = A'Y -> X = (A'A...)^-1 A'Y
        
        # Part 1: Left projection (Gene side)
        # We calculate (A.T @ A + Ridge) ^ -1 @ A.T @ Y
        # Using linalg.solve is more stable than explicit inverse
        AtA = A.T @ A
        reg_A = A_ridge * np.eye(AtA.shape[0])
        inv_part_A = np.linalg.solve(AtA + reg_A, A.T)
        
        # Part 2: Right projection (Perturbation side)
        # We need Y @ B @ (B.T @ B + Ridge)^-1
        BtB = B.T @ B
        reg_B = B_ridge * np.eye(BtB.shape[0])
        # Solve (BtB + Reg) * Z = B.T  => Z = (BtB...)^-1 B.T
        # Then transpose to get B (...) ^ -1
        inv_part_B = np.linalg.solve(BtB + reg_B, B.T).T 
        
        # Combine: LHS * Y * RHS
        K = inv_part_A @ Y_centered @ inv_part_B
        
    elif B is None:
        # Standard Ridge: Y = A * K
        AtA = A.T @ A
        reg_A = A_ridge * np.eye(AtA.shape[0])
        K = np.linalg.solve(AtA + reg_A, A.T @ Y_centered)
        
    elif A is None:
        # Y = K * B.T  => Y.T = B * K.T
        BtB = B.T @ B
        reg_B = B_ridge * np.eye(BtB.shape[0])
        K = (Y_centered @ np.linalg.solve(BtB + reg_B, B.T).T)
    else:
        raise ValueError("Either A or B must be non-null")

    # Handle NAs if any (simulating R's behavior)
    K = np.nan_to_num(K)
    
    return {"K": K, "center": center}

# ------------------------------------------------------------------------------
# 3. Load and Preprocess Data
# ------------------------------------------------------------------------------

folder = "data/gears_pert_data"
h5ad_path = os.path.join(folder, args.dataset_name, "perturb_processed.h5ad")
adata = sc.read_h5ad(h5ad_path)

# Load config
config_path = os.path.join(args.working_dir, "results", args.test_train_config_id)
with open(config_path, 'r') as f:
    set2condition = json.load(f)

# Ensure 'ctrl' is in training
if "ctrl" not in set2condition["train"]:
    set2condition["train"].append("ctrl")

# Create a flat list of all valid conditions
valid_conditions = set()
for k in set2condition:
    valid_conditions.update(set2condition[k])

# Filter adata
adata = adata[adata.obs['condition'].isin(valid_conditions)].copy()

# Clean conditions
adata.obs['clean_condition'] = adata.obs['condition'].astype(str).str.replace(r"\+ctrl", "", regex=True)

# Map conditions to training split
# Create a mapping dictionary: condition -> training_split
cond_to_split = {}
for split, conds in set2condition.items():
    for c in conds:
        cond_to_split[c] = split

adata.obs['training'] = adata.obs['condition'].map(cond_to_split)

# Identify gene names
gene_names = adata.var_names
# (R script sets rownames(sce) <- gene_names, which is standard in AnnData)

# Calculate Baseline (Mean of control)
ctrl_mask = adata.obs['condition'] == 'ctrl'
if sparse.issparse(adata.X):
    baseline = np.array(adata.X[ctrl_mask].mean(axis=0)).flatten()
else:
    baseline = np.mean(adata.X[ctrl_mask], axis=0)

# Pseudobulk
# Group by condition, clean_condition, training
# We aggregate to mean expression per condition
pseudobulk_obs = ['condition', 'clean_condition', 'training']
# Ensure these are strings/categories for grouping
pb_df = adata.to_df()
pb_df[pseudobulk_obs] = adata.obs[pseudobulk_obs].values

# Groupby and mean
psce_df = pb_df.groupby(pseudobulk_obs).mean()
psce_meta = pb_df[pseudobulk_obs].drop_duplicates().set_index('condition')

# Align metadata with the aggregated matrix
psce_df = psce_df.loc[psce_meta.index] # ensure alignment
psce_X = psce_df.values.T # Genes x Conditions (to match R's assay layout)

# Calculate "Change" (Y)
# Y = X_pseudobulk - baseline
Y_all = psce_X - baseline[:, None]

# Get Training Data Subset
train_mask = psce_meta['training'] == 'train'
train_conditions = psce_meta.index[train_mask]
Y_train = Y_all[:, train_mask] # Genes x TrainConditions

# ------------------------------------------------------------------------------
# 4. Generate Embeddings
# ------------------------------------------------------------------------------

# --- Gene Embeddings (A) ---
# Shape: Genes x pca_dim
if args.gene_embedding == "training_data":
    # PCA on Genes based on Training Data
    # Y_train is Genes x Conditions. 
    # sklearn PCA fits on (n_samples, n_features). Here Genes are observations.
    pca_op = PCA(n_components=args.pca_dim)
    # Fit on Genes (rows of Y_train)
    gene_emb = pca_op.fit_transform(Y_train) 
    gene_emb_rownames = psce_df.columns # Gene names
    
elif args.gene_embedding == "identity":
    gene_emb = np.eye(Y_train.shape[0])
    gene_emb_rownames = psce_df.columns
elif args.gene_embedding == "zero":
    gene_emb = np.zeros((Y_train.shape[0], Y_train.shape[0]))
    gene_emb_rownames = psce_df.columns
elif args.gene_embedding == "random":
    gene_emb = np.random.normal(size=(Y_train.shape[0], args.pca_dim))
    gene_emb_rownames = psce_df.columns
else:
    # Load from file
    # Assuming tsv with index as gene names
    df_emb = pd.read_csv(args.gene_embedding, sep="\t", index_col=0)
    gene_emb = df_emb.values.T # Transpose based on R script logic if needed, check dims
    # The R script does t(fread), implying the file is Pcs x Genes, so we want Genes x Pcs?
    # R: t(PCs x Genes) -> Genes x PCs. 
    # Python read_csv usually reads (Rows x Cols). If file is same format, just .values.
    # We'll assume the file is (Genes x PCs) standard, or transpose if shape matches.
    gene_emb = df_emb.values
    gene_emb_rownames = df_emb.index

# --- Perturbation Embeddings (B) ---
# Shape: Conditions x pca_dim (We need B, the R script passes B into solver)
if args.pert_embedding == "training_data":
    # R script: t(pca$x). This seems to define perturbation embedding based on
    # the expression profile of the condition.
    # We perform PCA on the Transpose of Y_train (Conditions x Genes)
    pca_op_pert = PCA(n_components=args.pca_dim)
    pert_emb = pca_op_pert.fit_transform(Y_train.T) # Conditions x PCs
    pert_emb_colnames = train_conditions # Condition names
    
elif args.pert_embedding == "identity":
    pert_emb = np.eye(Y_all.shape[1])
    pert_emb_colnames = psce_meta['clean_condition'].values
elif args.pert_embedding == "zero":
    pert_emb = np.zeros((Y_all.shape[1], Y_all.shape[1]))
    pert_emb_colnames = psce_meta['clean_condition'].values
elif args.pert_embedding == "random":
    # Random embedding for all conditions in psce
    # R script generates random for ncol(psce)
    pert_emb = np.random.normal(size=(Y_all.shape[1], args.pca_dim)) 
    pert_emb_colnames = psce_meta['clean_condition'].values
else:
    df_pert = pd.read_csv(args.pert_embedding, sep="\t", index_col=0)
    pert_emb = df_pert.values # Conditions x PCs
    pert_emb_colnames = df_pert.index

# Handle "ctrl" column requirement for external embeddings
# (If pert_emb is a dataframe from file, ensure ctrl exists)
if isinstance(pert_emb, pd.DataFrame) or (args.pert_embedding not in ["training_data", "identity", "zero", "random"]):
     # Logic to add zero row for ctrl if missing would go here, 
     # but for matrices (numpy) we rely on index matching below.
     pass

# ------------------------------------------------------------------------------
# 5. Matching and Alignment
# ------------------------------------------------------------------------------

# We need to subset Y, A, and B so dimensions match for the solver
# Y subset: Genes in gene_emb AND Conditions in pert_emb (subset to training)

# 1. Match Genes
common_genes = np.intersect1d(gene_emb_rownames, psce_df.columns)
if len(common_genes) <= 1:
    raise ValueError("Too few matches between gene names and gene_embedding")

gene_indices_Y = [psce_df.columns.get_loc(g) for g in common_genes]
gene_indices_A = [list(gene_emb_rownames).index(g) for g in common_genes]

Y_sub_genes = Y_train[gene_indices_Y, :]
A_sub = gene_emb[gene_indices_A, :]

# 2. Match Conditions (Perturbations)
# Note: In "training_data" mode, pert_emb was derived from Y_train, so it matches perfectly.
# In other modes, we need to match names.
clean_conds_train = psce_meta.loc[train_conditions, 'clean_condition'].values

# Find intersection between training conditions and available perturbation embeddings
common_conds_train = np.intersect1d(clean_conds_train, pert_emb_colnames)

if len(common_conds_train) <= 1:
     raise ValueError("Too few matches between clean_conditions and pert_embedding")

# Indices in Y_sub_genes (which is currently Genes x AllTrainConditions)
# We need to map clean_cond names back to the specific integer indices in Y_train
cond_indices_Y = []
cond_indices_B = []

# This loop ensures order is preserved
for c in common_conds_train:
    # Get index in Y_train (via train_conditions list which corresponds to columns)
    # Note: there might be multiple if duplicates exist, but pseudobulk is usually unique per cond
    # Assuming unique condition names in pseudobulk
    idx_y = np.where(clean_conds_train == c)[0][0] 
    cond_indices_Y.append(idx_y)
    
    # Get index in pert_emb
    idx_b = np.where(pert_emb_colnames == c)[0][0]
    cond_indices_B.append(idx_b)

Y_final = Y_sub_genes[:, cond_indices_Y]
B_training = pert_emb[cond_indices_B, :]

# ------------------------------------------------------------------------------
# 6. Solve Model
# ------------------------------------------------------------------------------

# Solve Y = A * K * B.T
# Note: pert_emb is usually (Conditions x PCs).
# The function expects B such that Y ~ A K B.T.
# If B_training is (N_conds x PCs), then B.T is (PCs x N_conds).
# Dimensions: (G x C) = (G x D1) * (D1 x D2) * (D2 x C).
# This matches.

solution = solve_y_axb(Y=Y_final, A=A_sub, B=B_training,
                       A_ridge=args.ridge_penalty, B_ridge=args.ridge_penalty)
K = solution['K']
center = solution['center']

# ------------------------------------------------------------------------------
# 7. Prediction
# ------------------------------------------------------------------------------

# Prepare B_all (Perturbation embedding for ALL conditions in the dataset)
# We map psce_meta['clean_condition'] to the embedding rows
pert_matches_all = []
valid_indices_all = []

all_clean_conds = psce_meta['clean_condition'].values
for i, c in enumerate(all_clean_conds):
    if c in pert_emb_colnames:
        idx_b = np.where(pert_emb_colnames == c)[0][0]
        pert_matches_all.append(pert_emb[idx_b, :])
        valid_indices_all.append(i)
    else:
        # Handle missing (e.g., zero fill or skip) - R script implies 0 if missing for ctrl
        if c == "ctrl":
            pert_matches_all.append(np.zeros(pert_emb.shape[1]))
            valid_indices_all.append(i)
        else:
            # If missing and not ctrl, we might just put zeros or fail. 
            # R script uses `match` which returns NA, then `na.omit` logic might drop them.
            # We will use zeros for missing perturbations to keep shape
            pert_matches_all.append(np.zeros(pert_emb.shape[1]))
            valid_indices_all.append(i)

B_all = np.array(pert_matches_all) # Conditions x PCs

# Baseline needs to be matched to the gene subset used
baseline_sub = baseline[gene_indices_Y]

# Pred = A * K * B_all.T + center + baseline
# Shapes: (G x D1) @ (D1 x D2) @ (D2 x C_all) -> (G x C_all)
pred = (A_sub @ K @ B_all.T) + center + baseline_sub[:, None]

# ------------------------------------------------------------------------------
# 8. Evaluation & Summary
# ------------------------------------------------------------------------------

# Calculate correlations (R2 in the R script context, but functionally Pearson)
# Observed change
obs = Y_all[gene_indices_Y, :] # Filtered genes, all conditions
# Note: obs is (Genes x Conditions), pred is (Genes x Conditions)

corrs = []
conditions_out = psce_meta.index
training_out = psce_meta['training']

print("Summary Statistics:")
for i in range(pred.shape[1]):
    # Flatten arrays for correlation
    p_vec = pred[:, i]
    o_vec = obs[:, i] + baseline_sub # Reconstruct raw expression or use change
    # R script compares `obs` (change) vs `pred - baseline` (change predicted)
    # Actually R script summary:
    # obs = assay(psce, "change")
    # pred = pred - baseline
    # It correlates the DELTAS.
    
    o_delta = obs[:, i] # This is already centered/change
    p_delta = pred[:, i] - baseline_sub
    
    if np.std(o_delta) == 0 or np.std(p_delta) == 0:
        r = 0
    else:
        r = np.corrcoef(o_delta, p_delta)[0, 1]
    
    corrs.append({
        "cond": conditions_out[i],
        "training": training_out[i],
        "r2": r
    })

df_res = pd.DataFrame(corrs)
print(df_res.groupby("training")['r2'].describe())

# ------------------------------------------------------------------------------
# 9. Save Results
# ------------------------------------------------------------------------------

# Format: JSON of dict(gene -> list of values? or list of dicts?)
# R script: as.list(as.data.frame(pred)) -> Column-oriented JSON (keys are Conditions)
# Pred matrix has Genes as Rows, Conditions as Columns.
# as.data.frame(pred) makes columns = conditions.
# as.list creates { "Cond1": [val1, val2...], "Cond2": ... }

pred_df = pd.DataFrame(pred, index=common_genes, columns=conditions_out)
output_dict = pred_df.to_dict(orient='list')

# Gene names list
output_genes = list(common_genes)

# Save to temp
tmp_dir = os.path.join(tempfile.gettempdir(), "prediction_storage")
os.makedirs(tmp_dir, exist_ok=True)

with open(os.path.join(tmp_dir, "all_predictions.json"), 'w') as f:
    json.dump(output_dict, f)

with open(os.path.join(tmp_dir, "gene_names.json"), 'w') as f:
    json.dump(output_genes, f)

# Move to final location
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.move(os.path.join(tmp_dir, "all_predictions.json"), os.path.join(out_dir, "all_predictions.json"))
shutil.move(os.path.join(tmp_dir, "gene_names.json"), os.path.join(out_dir, "gene_names.json"))

print(f"Done. Results saved to {out_dir}")