"""RDKit helpers shared by the generator, predictors and filters."""
from __future__ import annotations

import os
import sys

import numpy as np
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def to_mol(smiles: str | None) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles) if smiles else None


def canonical(smiles: str | None) -> str | None:
    mol = to_mol(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def basic_properties(mol: Chem.Mol) -> dict[str, float]:
    return {
        "logP": Crippen.MolLogP(mol),
        "QED": QED.qed(mol),
        "SAS": sascorer.calculateScore(mol),
        "MW": Descriptors.MolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def fingerprint(mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
    return _MORGAN.GetFingerprint(mol)


def _descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    values = []
    for _, fn in Descriptors.descList:
        try:
            values.append(fn(mol))
        except Exception:
            values.append(0.0)
    arr = np.nan_to_num(np.array(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1e6, 1e6).astype(np.float32)


def featurize(mol: Chem.Mol) -> np.ndarray:
    """Morgan count fingerprint (2048) concatenated with all RDKit 2D descriptors."""
    counts = _MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    return np.concatenate([counts, _descriptor_vector(mol)])
