"""End-to-end design loop: sample -> validate -> dedupe -> novelty -> safety -> predict -> rank."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs

from . import chem, safety
from .generator import Generator
from .predictor import PropertyPredictor


@dataclass
class DesignTargets:
    logp: float = 2.5       # desired Crippen logP
    qed: float = 0.8        # minimum drug-likeness (QED)
    sas: float = 3.0        # maximum synthetic-accessibility score (1 easy .. 10 hard)
    min_logs: float = -4.0  # minimum predicted aqueous solubility, logS (mol/L)


@dataclass
class DesignResult:
    table: pd.DataFrame
    stats: dict


def desirability(df: pd.DataFrame, t: DesignTargets) -> np.ndarray:
    """Geometric mean of per-property desirabilities in [0, 1], discounted per Brenk alert."""
    d_logp = np.exp(-0.5 * ((df["logP"] - t.logp) / 0.75) ** 2)
    d_qed = np.where(df["QED"] >= t.qed, 1.0, np.exp(-0.5 * ((t.qed - df["QED"]) / 0.1) ** 2))
    d_sas = np.where(df["SAS"] <= t.sas, 1.0, np.exp(-0.5 * ((df["SAS"] - t.sas) / 0.5) ** 2))
    d_sol = 1.0 / (1.0 + np.exp(-(df["pred_logS"] - t.min_logs) / 0.4))
    geo = np.cbrt(d_logp * d_qed * d_sas) ** 0.75 * d_sol ** 0.25
    return geo * 0.9 ** df["brenk_alerts"]


class DesignPipeline:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.generator = Generator.load(model_dir / "generator.pt")
        self.solubility = PropertyPredictor.load(model_dir / "solubility.joblib")
        self.logd = PropertyPredictor.load(model_dir / "logd.joblib")
        self.train_smiles = set((model_dir / "train_smiles.txt").read_text().splitlines())

    def design(self, targets: DesignTargets, n_samples: int = 2000, top_k: int = 20,
               temperature: float = 1.0, max_similarity: float = 0.7,
               seed: int | None = None) -> DesignResult:
        if seed is not None:
            torch.manual_seed(seed)
        raw =self.generator.sample(targets.logp, targets.qed, targets.sas, n=n_samples,
                                    temperature=temperature)

        unique: dict[str, Chem.Mol] = {}
        n_valid = 0
        for smiles in raw:
            mol = chem.to_mol(smiles)
            if mol is None:
                continue
            n_valid += 1
            unique.setdefault(Chem.MolToSmiles(mol), mol)

        novel = {s: m for s, m in unique.items() if s not in self.train_smiles}
        accepted: dict[str, tuple[Chem.Mol, safety.Assessment]] = {}
        for smiles, mol in novel.items():
            verdict = safety.assess(mol)
            if verdict.passed:
                accepted[smiles] = (mol, verdict)

        stats = {"sampled": len(raw), "valid": n_valid, "unique": len(unique), "novel": len(novel),
                 "passed_safety": len(accepted)}
        if not accepted:
            return DesignResult(pd.DataFrame(), stats)

        smiles_list = list(accepted)
        mols = [accepted[s][0] for s in smiles_list]
        logs, logs_sd = self.solubility.predict(mols)
        logd, logd_sd = self.logd.predict(mols)
        df = pd.DataFrame([chem.basic_properties(m) for m in mols])
        df.insert(0, "SMILES", smiles_list)
        df["pred_logS"], df["logS_std"] = logs, logs_sd
        df["pred_logD"], df["logD_std"] = logd, logd_sd
        df["brenk_alerts"] = [accepted[s][1].brenk_alerts for s in smiles_list]
        df.insert(1, "score", desirability(df, targets))
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        return DesignResult(self._diverse_top(df, mols_by_smiles=dict(zip(smiles_list, mols)),
                                              top_k=top_k, max_similarity=max_similarity), stats)

    @staticmethod
    def _diverse_top(df: pd.DataFrame, mols_by_smiles: dict[str, Chem.Mol], top_k: int,
                     max_similarity: float) -> pd.DataFrame:
        """Walk down the ranking, skipping near-duplicates (Tanimoto >= max_similarity)."""
        chosen: list[int] = []
        chosen_fps = []
        for i, smiles in enumerate(df["SMILES"]):
            fp = chem.fingerprint(mols_by_smiles[smiles])
            if chosen_fps and max(DataStructs.BulkTanimotoSimilarity(fp, chosen_fps)) >= max_similarity:
                continue
            chosen.append(i)
            chosen_fps.append(fp)
            if len(chosen) == top_k:
                break
        return df.iloc[chosen].reset_index(drop=True)
