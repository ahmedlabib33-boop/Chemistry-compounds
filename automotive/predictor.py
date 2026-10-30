"""Property predictors for automotive-relevant physical properties, reusing chemgen's random
forest architecture (Morgan counts + RDKit descriptors, scaffold-split evaluation, tree-spread
uncertainty) against the OPERA measurements in automotive/data.py.

Four targets, each with real, measured, scaffold-split-evaluated training data:
    BP  boiling point, degrees C          -- fuel volatility / distillation range
    VP  vapor pressure, log10(mmHg)       -- evaporative emissions; tracks flammability with BP
    MP  melting point, degrees C          -- cold-flow / pour-point proxy for fuels and lubricants
    WS  water solubility, log10(mol/L)    -- miscibility with coolant/moisture, spill persistence

What this does NOT predict, and why: flash point and kinematic viscosity are the two properties
an automotive chemist would ask for first, but no open dataset of either was found that can be
downloaded programmatically (OPERA does not include them; nothing comparable to ZINC/ESOL exists
for them). Fabricating a "flash point model" from a textbook correlation to boiling point would
overstate what is actually known here, so this app reports boiling point and vapor pressure as
measured/predicted quantities and leaves flash point as something the lower of the two roughly
tracks, stated qualitatively, not computed.
"""
from __future__ import annotations

from chemgen.predictor import DatasetSpec, PropertyPredictor  # re-exported for callers

SMILES_COL, TARGET_COL = "Canonical_QSARr", "value_point_estimate"

DATASETS = {
    "bp": DatasetSpec("bp.csv", SMILES_COL, TARGET_COL, "boiling point, degrees C"),
    "vp": DatasetSpec("vp.csv", SMILES_COL, TARGET_COL, "vapor pressure, log10(mmHg) at 25C"),
    "mp": DatasetSpec("mp.csv", SMILES_COL, TARGET_COL, "melting point, degrees C"),
    "ws": DatasetSpec("ws.csv", SMILES_COL, TARGET_COL, "water solubility, log10(mol/L)"),
}


def train(name, data_dir, **kwargs) -> tuple[PropertyPredictor, dict]:
    return PropertyPredictor.train(name, data_dir, specs=DATASETS, **kwargs)
