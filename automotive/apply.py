"""Apply the trained automotive models to a structure you supply: screen it, predict its boiling
point, vapor pressure, melting point and water solubility with an applicability check, tag it
with any matching automotive-additive roles, and attach a preparation recipe.

Same three tiers as chemgen.apply: refused (hazard screen) / unsupported (unparseable, disallowed
elements, radicals) / ok (predictions given; a recipe is withheld if the structure carries NIH
reactive/toxic alerts).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs

from chemgen import chem, multistep

from . import roles, safety
from .predictor import DATASETS

IN_DOMAIN_SIMILARITY = 0.35
HARD_PREFIXES = ("hazard screen", "disallowed elements", "radical")


@dataclass
class ModelBundle:
    predictors: dict
    domain_fps: dict = field(default_factory=dict)  # name -> list[fingerprint] of that predictor's training set

    @classmethod
    def load(cls, model_dir: Path, data_dir: Path) -> "ModelBundle":
        from chemgen.predictor import PropertyPredictor
        model_dir, data_dir = Path(model_dir), Path(data_dir)
        predictors, fps = {}, {}
        for name, spec in DATASETS.items():
            predictors[name] = PropertyPredictor.load(model_dir / f"{name}.joblib")
            path = data_dir / spec.filename
            if path.exists():
                column = pd.read_csv(path, usecols=[spec.smiles_col])[spec.smiles_col]
                fps[name] = [chem.fingerprint(m) for m in map(chem.to_mol, column) if m is not None]
        return cls(predictors, fps)

    def nearest_similarity(self, mol: Chem.Mol, name: str) -> float | None:
        pool = self.domain_fps.get(name)
        if not pool:
            return None
        return max(DataStructs.BulkTanimotoSimilarity(chem.fingerprint(mol), pool))


@dataclass
class Analysis:
    input: str
    status: str
    smiles: str | None = None
    reasons: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)       # name -> (mean, std, nearest_similarity)
    role_matches: list = field(default_factory=list)
    recipe: multistep.MultiStepRecipe | None = None
    recipe_note: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        head = f"## `{self.smiles or self.input}`"
        if self.status == "invalid":
            return f"{head}\n\nCould not read this as a SMILES string."
        if self.status == "refused":
            return (f"{head}\n\n**Refused.** This structure matches ChemGen's hazard screen "
                    f"({'; '.join(r.split(': ', 1)[-1] for r in self.reasons)}). No predictions or "
                    "preparation are provided for it.")
        if self.status == "unsupported":
            return f"{head}\n\n**Not supported:** {'; '.join(self.reasons)}."

        p = self.properties
        lines = [head, "", "**Computed properties (RDKit)**", "",
                 "| logP | TPSA | SA score | MW |", "| --- | --- | --- | --- |",
                 f"| {p['logP']:.2f} | {p['TPSA']:.1f} | {p['SAS']:.2f} | {p['MW']:.1f} |", ""]
        labels = {"bp": "Boiling point (degrees C)", "vp": "Vapor pressure (log10 mmHg)",
                  "mp": "Melting point (degrees C)", "ws": "Water solubility (log10 mol/L)"}
        lines += ["**ML predictions** (random forests trained on EPA OPERA measurements; "
                  "+/- is the spread across trees)", ""]
        for name, label in labels.items():
            mean, std, sim = self.predictions[name]
            domain = "" if sim is None else (
                ", within the model's training domain" if sim >= IN_DOMAIN_SIMILARITY
                else f", FAR from the model's training data (nearest-neighbour Tanimoto {sim:.2f}): "
                     "treat this one as unreliable")
            lines.append(f"- {label}: {mean:.2f} +/- {std:.2f}{domain}")
        lines.append("")
        lines.append("Flash point and viscosity are not predicted: no open, downloadable dataset for either "
                     "was found (see automotive/predictor.py). Flash point roughly tracks boiling point and "
                     "vapor pressure (lower BP / higher VP -> more flammable), stated qualitatively only.")
        if self.role_matches:
            lines += ["", "**Automotive role (rule-based functional-group match, not a measured or learned "
                          "activity)**", ""]
            lines += [f"- {m.label}: {m.note}" for m in self.role_matches]
        else:
            lines += ["", "**Automotive role:** no match against the rule-based role patterns "
                          "(automotive.roles) -- that does not mean it has no automotive use."]
        if self.notes:
            lines += ["", "**Notes**", *[f"- {n}" for n in self.notes]]
        lines.append("")
        if self.recipe:
            lines += ["**How to prepare it**", "", self.recipe.to_markdown()]
        else:
            lines.append(f"**How to prepare it:** {self.recipe_note}")
        return "\n".join(lines)


def analyze(smiles: str, bundle: ModelBundle, scale_mmol: float = 1.0, max_steps: int = 3) -> Analysis:
    mol = chem.to_mol((smiles or "").strip())
    if mol is None:
        return Analysis(input=smiles, status="invalid")
    mol = chem.neutralize(mol)
    canonical = Chem.MolToSmiles(mol)

    verdict = safety.assess(mol)
    hazard = [r for r in verdict.reasons if r.startswith("hazard screen")]
    if hazard:
        return Analysis(input=smiles, status="refused", reasons=hazard)
    hard = [r for r in verdict.reasons if r.startswith(HARD_PREFIXES) or r == "net charge"]
    if hard:
        return Analysis(input=smiles, status="unsupported", smiles=canonical, reasons=hard)

    analysis = Analysis(input=smiles, status="ok", smiles=canonical, properties=chem.basic_properties(mol),
                        role_matches=roles.classify(mol))
    for name, predictor in bundle.predictors.items():
        mean, std = predictor.predict([mol])
        analysis.predictions[name] = (float(mean[0]), float(std[0]), bundle.nearest_similarity(mol, name))

    analysis.notes = [f"outside the generator's design space: {r}" for r in verdict.reasons]
    toxic = safety.nih_alerts(mol)
    if toxic:
        analysis.recipe_note = ("withheld: the structure carries reactive or toxic structural alerts "
                                f"({', '.join(sorted(set(toxic)))}).")
    else:
        analysis.recipe = multistep.propose_plan(mol, scale_mmol, max_steps)
        if analysis.recipe is None:
            analysis.recipe_note = ("none of the nine supported reaction classes disconnects this "
                                    "structure within the step limit; it needs a route designed "
                                    "by a chemist.")
    return analysis
