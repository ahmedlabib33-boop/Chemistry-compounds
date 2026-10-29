"""Property predictors: random forests on Morgan counts + RDKit descriptors.

The spread across trees is reported as an uncertainty estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from . import chem


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    smiles_col: str
    target_col: str
    label: str


DATASETS = {
    "solubility": DatasetSpec("delaney.csv", "smiles", "measured log solubility in mols per litre",
                              "aqueous solubility, logS (mol/L)"),
    "logd": DatasetSpec("lipophilicity.csv", "smiles", "exp", "lipophilicity, logD at pH 7.4"),
}


def _scaffold_split(mols: list[Chem.Mol], test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic scaffold split: whole scaffolds go to train or test, never both."""
    groups: dict[str, list[int]] = {}
    for i, mol in enumerate(mols):
        groups.setdefault(MurckoScaffold.MurckoScaffoldSmiles(mol=mol), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_test, test = int(len(mols) * test_frac), []
    train: list[int] = []
    for group in ordered:
        (train if len(train) + len(group) <= len(mols) - n_test else test).extend(group)
    return np.array(train), np.array(test)


class PropertyPredictor:
    def __init__(self, name: str, model: RandomForestRegressor):
        self.name, self.model = name, model

    @classmethod
    def train(cls, name: str, data_dir: Path, n_estimators: int = 400,
              seed: int = 0) -> tuple["PropertyPredictor", dict]:
        spec = DATASETS[name]
        df = pd.read_csv(Path(data_dir) / spec.filename)
        mols = [chem.to_mol(s) for s in df[spec.smiles_col]]
        ok = [m is not None for m in mols]
        mols = [m for m in mols if m is not None]
        y = df.loc[ok, spec.target_col].to_numpy(dtype=np.float64)
        X = np.stack([chem.featurize(m) for m in mols])

        def forest() -> RandomForestRegressor:
            return RandomForestRegressor(n_estimators=n_estimators, max_features=0.33,
                                         n_jobs=-1, random_state=seed)

        train_idx, test_idx = _scaffold_split(mols)
        held_out = forest().fit(X[train_idx], y[train_idx])
        pred = held_out.predict(X[test_idx])
        metrics = {
            "target": spec.label,
            "n_molecules": len(mols),
            "n_test": len(test_idx),
            "rmse": float(root_mean_squared_error(y[test_idx], pred)),
            "mae": float(mean_absolute_error(y[test_idx], pred)),
            "r2": float(r2_score(y[test_idx], pred)),
            "baseline_rmse": float(root_mean_squared_error(y[test_idx], np.full(len(test_idx),
                                                                                 y[train_idx].mean()))),
        }
        return cls(name, forest().fit(X, y)), metrics

    def predict(self, mols: list[Chem.Mol]) -> tuple[np.ndarray, np.ndarray]:
        X = np.stack([chem.featurize(m) for m in mols])
        per_tree = np.stack([tree.predict(X) for tree in self.model.estimators_])
        return per_tree.mean(0), per_tree.std(0)

    def save(self, path: Path) -> None:
        joblib.dump({"name": self.name, "model": self.model}, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "PropertyPredictor":
        blob = joblib.load(path)
        return cls(blob["name"], blob["model"])
