"""Builds the training CSV for the automotive generator and trains it, reusing
chemgen.generator.train_generator and chemgen.generator.Generator UNCHANGED.

chemgen's ConditionalSmilesLM conditions on three floats; it has no idea what they represent,
so the same architecture and the same train_generator()/Generator.sample(a, b, c, ...) call
works for a different domain just by writing a training CSV with different numbers under the
same three column names ("smiles", "logP", "qed", "SAS") that function expects. Here the "qed"
column holds TPSA, not QED -- this module is the one place that doc exists; everywhere else
(automotive/pipeline.py, CLI, UI) calls it by its real name, TPSA.

Training pool: every molecule from automotive/data.py's four OPERA property files (boiling
point, vapor pressure, melting point, water solubility) that passes automotive.safety's element
and molecular-weight screen -- a broad, non-drug-biased set of real industrial/environmental
chemicals, a much better match for automotive chemistry than a drug-discovery set like ZINC.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from chemgen import chem
from chemgen.generator import train_generator as _train_generator

from . import safety
from .predictor import DATASETS

RDLogger.DisableLog("rdApp.*")


def build_training_table(data_dir: Path) -> pd.DataFrame:
    smiles: set[str] = set()
    for spec in DATASETS.values():
        path = Path(data_dir) / spec.filename
        if path.exists():
            smiles.update(pd.read_csv(path, usecols=[spec.smiles_col])[spec.smiles_col])

    rows = []
    seen = set()
    for raw in smiles:
        mol = chem.to_mol(raw)
        if mol is None or not safety.assess(mol).passed:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        props = chem.basic_properties(mol)
        rows.append({"smiles": canonical, "logP": props["logP"], "qed": props["TPSA"], "SAS": props["SAS"]})
    return pd.DataFrame(rows)


def train_generator(data_dir: Path, out_path: Path, train_smiles_path: Path, training_csv_path: Path,
                    n_mols: int = 20_000, epochs: int = 10, seed: int = 0, log=print) -> dict:
    table = build_training_table(data_dir)
    log(f"automotive training pool: {len(table)} molecules (boiling point / vapor pressure / "
        f"melting point / water solubility sources, after the automotive safety screen)")
    training_csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(training_csv_path, index=False)
    return _train_generator(training_csv_path, out_path, train_smiles_path, n_mols=n_mols,
                            epochs=epochs, seed=seed, log=log)
