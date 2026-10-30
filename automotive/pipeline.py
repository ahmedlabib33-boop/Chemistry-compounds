"""End-to-end design loop for automotive chemicals: sample -> validate -> dedupe -> novelty ->
safety -> predict physical properties -> tag roles -> rank -> plan synthesis.

Reuses, unchanged, from chemgen: the SMILES tokenizer and the conditional generator
architecture (chemgen.generator.Generator; see automotive/generator.py for the column-reuse
note), the hazard screen and NIH alert catalog (via automotive.safety), the diversity-aware
top-k selection (chemgen.pipeline.DesignPipeline._diverse_top) and the multi-step retrosynthesis
planner (chemgen.multistep). Only the domain-specific pieces are new: automotive.safety's wider
element whitelist, automotive.predictor's physical-property models, and automotive.roles's
functional-group tagging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from chemgen import chem, multistep
from chemgen.generator import Generator
from chemgen.pipeline import DesignPipeline as _DrugPipeline
from chemgen.predictor import PropertyPredictor

from . import roles, safety
from .predictor import DATASETS

# condition slots the shared Generator architecture is trained on for this domain -- see
# automotive/generator.py: the GRU doesn't know or care what the three floats "mean".
CONDITION_LABELS = ("logP", "TPSA", "SAS")


@dataclass
class AutomotiveTargets:
    logp: float = 4.0        # target Crippen logP (automotive chemicals are usually more
                              # lipophilic than drugs: base oils, fuels and most additives
                              # need to dissolve in a nonpolar hydrocarbon phase)
    tpsa: float = 20.0        # target topological polar surface area (low = base-oil/fuel-like,
                              # higher = a more polar functional additive)
    sas: float = 3.0          # maximum synthetic-accessibility score
    role: str | None = None   # restrict to this automotive role (see automotive.roles.ROLES); None = any


@dataclass
class DesignResult:
    table: pd.DataFrame
    stats: dict
    recipes: dict[str, multistep.MultiStepRecipe | None] = field(default_factory=dict)

    def recipes_markdown(self) -> str:
        parts = []
        for i, smiles in enumerate(self.table["SMILES"], 1):
            recipe = self.recipes.get(smiles)
            body = recipe.to_markdown() if recipe else "No route from the supported reaction classes within the step limit."
            parts.append(f"## Molecule {i}: `{smiles}`\n\n{body}")
        return "\n\n---\n\n".join(parts)


def desirability(df: pd.DataFrame, t: AutomotiveTargets) -> np.ndarray:
    d_logp = np.exp(-0.5 * ((df["logP"] - t.logp) / 1.5) ** 2)
    d_tpsa = np.exp(-0.5 * ((df["TPSA"] - t.tpsa) / 20.0) ** 2)
    d_sas = np.where(df["SAS"] <= t.sas, 1.0, np.exp(-0.5 * ((df["SAS"] - t.sas) / 0.5) ** 2))
    geo = (d_logp * d_tpsa * d_sas) ** (1 / 3)
    if t.role is None:
        return geo
    matched = df["roles"].map(lambda tags: t.role in tags)
    return geo * np.where(matched, 1.0, 0.5)   # a role filter narrows the table; it doesn't hard-exclude


class AutomotivePipeline:
    def __init__(self, model_dir: Path, data_dir: Path):
        model_dir, data_dir = Path(model_dir), Path(data_dir)
        self.generator = Generator.load(model_dir / "generator.pt")
        self.predictors = {name: PropertyPredictor.load(model_dir / f"{name}.joblib") for name in DATASETS}
        self.train_smiles = set((model_dir / "train_smiles.txt").read_text().splitlines())

    def design(self, targets: AutomotiveTargets, n_samples: int = 2000, top_k: int = 20,
               temperature: float = 1.0, max_similarity: float = 0.7, seed: int | None = None,
               max_steps: int = 3, with_recipes: bool = True) -> DesignResult:
        import torch
        if seed is not None:
            torch.manual_seed(seed)
        raw = self.generator.sample(targets.logp, targets.tpsa, targets.sas, n=n_samples, temperature=temperature)

        unique: dict[str, Chem.Mol] = {}
        n_valid = 0
        for smiles in raw:
            mol = chem.to_mol(smiles)
            if mol is None:
                continue
            n_valid += 1
            mol = chem.neutralize(mol)
            unique.setdefault(Chem.MolToSmiles(mol), mol)

        novel = {s: m for s, m in unique.items() if s not in self.train_smiles}
        accepted: dict[str, Chem.Mol] = {}
        for smiles, mol in novel.items():
            if safety.assess(mol).passed:
                accepted[smiles] = mol

        stats = {"sampled": len(raw), "valid": n_valid, "unique": len(unique), "novel": len(novel),
                 "passed_safety": len(accepted)}
        if not accepted:
            return DesignResult(pd.DataFrame(), stats)

        smiles_list = list(accepted)
        mols = [accepted[s] for s in smiles_list]
        df = pd.DataFrame([chem.basic_properties(m) for m in mols])
        df.insert(0, "SMILES", smiles_list)
        for name, spec in DATASETS.items():
            mean, std = self.predictors[name].predict(mols)
            df[f"pred_{name}"], df[f"{name}_std"] = mean, std
        df["roles"] = [[m.key for m in roles.classify(mol)] for mol in mols]
        df.insert(1, "score", desirability(df, targets))
        df["role_tags"] = df["roles"].map(lambda tags: ", ".join(tags) or "-")
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        table = _DrugPipeline._diverse_top(df, mols_by_smiles=dict(zip(smiles_list, mols)),
                                           top_k=top_k, max_similarity=max_similarity)
        recipes: dict[str, multistep.MultiStepRecipe | None] = {}
        if with_recipes:
            by_smiles = dict(zip(smiles_list, mols))
            recipes = {s: multistep.propose_plan(by_smiles[s], max_steps=max_steps) for s in table["SMILES"]}
        return DesignResult(table, stats, recipes)
