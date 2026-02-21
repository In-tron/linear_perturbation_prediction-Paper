import json
import re
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Args -------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Collect ground truth for combinatorial data')
parser.add_argument('--test_train_config_id', required=True)
parser.add_argument('--working_dir', required=True)
args = parser.parse_args()
print(args)

# --- Load train/test split --------------------------------------------------

with open(Path(args.working_dir) / "results" / args.test_train_config_id) as f:
    raw = json.load(f)

test_train = (
    pd.DataFrame([(split, pert) for split, perts in raw.items() for pert in perts],
                 columns=["train", "perturbation"])
)

# --- Helpers ----------------------------------------------------------------

def normalize_perturbation(name: str) -> str:
    """Replicate the R mutate logic: standardize condition name to 'A+B' form."""
    parts = re.split(r"[+_]", name, maxsplit=1)
    parts = [p for p in parts if p not in ("ctrl", "")]
    if not parts:
        return "ctrl"
    if len(parts) == 1:
        parts.append("ctrl")
    return "+".join(parts)

def load_results(path) -> pd.DataFrame:
    """Load predictions from a result directory and return a DataFrame with columns: perturbation, gene, prediction."""
    path = Path(path)
    with open(path / "all_predictions.json") as f:
        preds = json.load(f)
    with open(path / "gene_names.json") as f:
        genes = json.load(f)

    rows = []
    for pert, values in preds.items():
        norm = normalize_perturbation(pert)
        for gene, val in zip(genes, values):
            rows.append((norm, gene, val))

    return pd.DataFrame(rows, columns=["perturbation", "gene", "prediction"])


# --- Load data --------------------------------------------------------------

ground_truth = load_results(Path(args.working_dir) / "results" / "ground_truth_results/").rename(
    columns={"prediction": "truth"}
)

preds = pd.concat(
    {
        "linear": load_results(Path(args.working_dir) / "results" / "linear_results/"),
        "mean":   load_results(Path(args.working_dir) / "results" / "mean_results/"),
        "gears":  load_results(Path(args.working_dir) / "results" / "gears_results/"),
        # "scGPT":  load_results(Path(args.working_dir) / "results" / "sgpt_results/"),
    },
    names=["method"]
).reset_index(level="method").reset_index(drop=True)

# --- Top 1000 highest-expressed genes at ctrl -------------------------------

highest_expr_genes = (
    ground_truth[ground_truth["perturbation"] == "ctrl"]
    .nlargest(1000, "truth")
    [["gene", "truth"]]
    .rename(columns={"truth": "baseline"})
)

# --- Compute metrics --------------------------------------------------------

res = (
    preds
    .merge(ground_truth, on=["gene", "perturbation"])
    .merge(highest_expr_genes, on="gene")
    .groupby(["perturbation", "method"])
    .apply(lambda g: pd.Series({
        "dist":          np.sqrt(((g["truth"] - g["prediction"]) ** 2).sum()),
        "pearson_delta": np.corrcoef(
                             g["prediction"] - g["baseline"],
                             g["truth"]      - g["baseline"]
                         )[0, 1],
    }), include_groups=False)
    .reset_index()
)

# --- Plot -------------------------------------------------------------------

plot_df = (
    res
    .merge(test_train, on="perturbation")
    .query("train != 'train'")
)

fig, ax = plt.subplots(figsize=(8, 5))
sns.stripplot(data=plot_df, x="method", y="dist", ax=ax, jitter=True, alpha=0.6)
ax.set_xlabel("Method")
ax.set_ylabel("Distance")
ax.set_title("Prediction distance by method (test/val conditions)")
plt.tight_layout()
plt.show()