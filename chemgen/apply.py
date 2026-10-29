"""Apply the trained models to a structure you supply: screen it, predict its properties with
an applicability check, and write a preparation recipe.

Three tiers, so the tool stays useful without becoming a route-writer for anything at all:

* ``refused``     matches the hazard screen (controlled-substance / energetic / mustard
                  patterns). Nothing is predicted or written.
* ``unsupported`` cannot be handled (unparseable, elements outside C N O S F Cl Br I, radicals).
* ``ok``          predictions are given. A preparation recipe is written unless the structure
                  carries NIH reactive/toxic alerts, in which case the recipe is withheld.

Molecular weight, Lipinski and PAINS limits that shape the *generator* are reported as notes
here rather than blocking, because you may want an answer for an ordinary small molecule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs

from . import DATA_DIR, MODEL_DIR, chem, multistep, safety
from .predictor import DATASETS, PropertyPredictor

HARD_PREFIXES = ("hazard screen", "disallowed elements", "radical")
IN_DOMAIN_SIMILARITY = 0.35  # below this the property models have seen nothing similar


@dataclass
class ModelBundle:
    solubility: PropertyPredictor
    logd: PropertyPredictor
    domain_fps: list = field(default_factory=list)  # fingerprints of the property models' training data

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR, data_dir: Path = DATA_DIR) -> "ModelBundle":
        fps = []
        for spec in DATASETS.values():
            path = Path(data_dir) / spec.filename
            if path.exists():
                column = pd.read_csv(path, usecols=[spec.smiles_col])[spec.smiles_col]
                fps += [chem.fingerprint(m) for m in map(chem.to_mol, column) if m is not None]
        model_dir = Path(model_dir)
        return cls(PropertyPredictor.load(model_dir / "solubility.joblib"),
                   PropertyPredictor.load(model_dir / "logd.joblib"), fps)

    def nearest_similarity(self, mol: Chem.Mol) -> float | None:
        if not self.domain_fps:
            return None
        return max(DataStructs.BulkTanimotoSimilarity(chem.fingerprint(mol), self.domain_fps))


@dataclass
class Analysis:
    input: str
    status: str  # ok | refused | unsupported | invalid
    smiles: str | None = None
    reasons: list[str] = field(default_factory=list)
    properties: dict[str, float] = field(default_factory=dict)
    predictions: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (mean, std)
    nearest_similarity: float | None = None
    recipe: multistep.MultiStepRecipe | None = None
    recipe_note: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def in_domain(self) -> bool | None:
        if self.nearest_similarity is None:
            return None
        return self.nearest_similarity >= IN_DOMAIN_SIMILARITY

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

        p, lines = self.properties, [head, ""]
        lines += ["**Computed properties (RDKit)**", "",
                  "| logP | QED | SA score | MW | TPSA | HBD | HBA | Rot. bonds |", "| --- | --- | --- | --- | --- | --- | --- | --- |",
                  f"| {p['logP']:.2f} | {p['QED']:.2f} | {p['SAS']:.2f} | {p['MW']:.1f} | {p['TPSA']:.1f} | "
                  f"{p['HBD']:.0f} | {p['HBA']:.0f} | {p['RotB']:.0f} |", ""]
        s_mean, s_std = self.predictions["logS"]
        d_mean, d_std = self.predictions["logD"]
        grams_per_litre = 10 ** s_mean * p["MW"]
        lines += ["**ML predictions** (random forests; ± is the spread across trees)", "",
                  f"- Aqueous solubility logS = {s_mean:.2f} ± {s_std:.2f} mol/L (about {grams_per_litre:.2g} g/L)",
                  f"- Lipophilicity logD(7.4) = {d_mean:.2f} ± {d_std:.2f}", ""]
        if self.nearest_similarity is not None:
            verdict = ("within the models' training domain" if self.in_domain
                       else "FAR from the models' training data: treat the predictions as unreliable")
            lines += [f"**Applicability:** nearest training neighbour Tanimoto {self.nearest_similarity:.2f}, "
                      f"{verdict}.", ""]
        if self.notes:
            lines += ["**Notes**", *[f"- {n}" for n in self.notes], ""]
        if self.recipe:
            lines += ["**How to prepare it**", "", self.recipe.to_markdown()]
        else:
            lines += [f"**How to prepare it:** {self.recipe_note}"]
        return "\n".join(lines)


def analyze(smiles: str, bundle: ModelBundle, scale_mmol: float = 1.0, max_steps: int = 3,
            known_lookup=None) -> Analysis:
    mol = chem.to_mol((smiles or "").strip())
    if mol is None:
        return Analysis(input=smiles, status="invalid")
    mol = chem.standardize(mol)
    canonical = Chem.MolToSmiles(mol)

    verdict = safety.assess(mol)
    hazard = [r for r in verdict.reasons if r.startswith("hazard screen")]
    if hazard:
        return Analysis(input=smiles, status="refused", reasons=hazard)
    hard = [r for r in verdict.reasons if r.startswith(HARD_PREFIXES) or r == "net charge"]
    if hard:
        return Analysis(input=smiles, status="unsupported", smiles=canonical, reasons=hard)

    analysis = Analysis(input=smiles, status="ok", smiles=canonical,
                        properties=chem.basic_properties(mol),
                        nearest_similarity=bundle.nearest_similarity(mol))
    for name, predictor in (("logS", bundle.solubility), ("logD", bundle.logd)):
        mean, std = predictor.predict([mol])
        analysis.predictions[name] = (float(mean[0]), float(std[0]))

    analysis.notes = [f"outside the generator's design space: {r}" for r in verdict.reasons]
    if (analysis.nearest_similarity or 0) >= 0.999:
        analysis.notes.append("this exact structure is in the property models' training data, so the "
                              "predictions are a fit, not a held-out estimate")
    if verdict.brenk_alerts:
        analysis.notes.append(f"{verdict.brenk_alerts} Brenk alert(s) for unwanted functionality")

    toxic = safety.nih_alerts(mol)
    if toxic:
        analysis.recipe_note = ("withheld: the structure carries reactive or toxic structural alerts "
                                f"({', '.join(sorted(set(toxic)))}).")
    else:
        analysis.recipe = multistep.propose_plan(mol, scale_mmol, max_steps)
        if analysis.recipe is None:
            analysis.recipe_note = ("none of the nine supported reaction classes disconnects this "
                                    "structure; it needs a route designed by a chemist.")
        elif known_lookup is not None:
            analysis.recipe.annotate(known_lookup)
    return analysis
