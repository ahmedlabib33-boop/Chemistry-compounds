"""Screening for the automotive domain. Reuses chemgen.safety's hazard screen (controlled
substances, energetic groups, mustard agents -- these never depend on the domain) and its NIH
reactive/toxic alert catalog, but replaces the drug-specific rules: the element whitelist includes
phosphorus and boron (organophosphate antiwear additives and borate-ester friction modifiers are
ordinary automotive chemistry), there is no Lipinski/drug-likeness gate, and the molecular-weight
window is widened to cover base-oil-range molecules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from chemgen import safety as drug_safety

ALLOWED_ELEMENTS = frozenset({"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"})
MW_RANGE = (40.0, 700.0)


@dataclass
class Assessment:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def assess(mol: Chem.Mol) -> Assessment:
    reasons: list[str] = []

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    disallowed = elements - ALLOWED_ELEMENTS
    if disallowed:
        reasons.append(f"disallowed elements: {','.join(sorted(disallowed))}")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        reasons.append("radical")
    if len(Chem.GetMolFrags(mol)) > 1:
        reasons.append("multiple fragments")
    if Chem.GetFormalCharge(mol) != 0:
        reasons.append("net charge")

    from rdkit.Chem import Descriptors
    mw = Descriptors.MolWt(mol)
    if not MW_RANGE[0] <= mw <= MW_RANGE[1]:
        reasons.append(f"molecular weight {mw:.0f} outside {MW_RANGE}")

    reasons += [f"hazard screen: {flag}" for flag in drug_safety.hazard_flags(mol)]
    return Assessment(passed=not reasons, reasons=reasons)


def nih_alerts(mol: Chem.Mol) -> list[str]:
    return drug_safety.nih_alerts(mol)
