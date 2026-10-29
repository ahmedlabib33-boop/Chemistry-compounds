"""Hard filters applied to every generated molecule before it is shown to a user.

The generator is trained on drug-like ZINC molecules, but a model can still emit
off-distribution structures. These checks keep results inside a drug-like,
non-reactive, non-toxic-alert space regardless of what the sampler produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Elements that occur in ordinary drug-like molecules. Anything else is rejected.
ALLOWED_ELEMENTS = frozenset({"C", "N", "O", "S", "F", "Cl", "Br", "I"})

MW_RANGE = (150.0, 600.0)
MAX_LIPINSKI_VIOLATIONS = 1


# Structural patterns for controlled-substance classes, explosives and energetic groups.
# This is a coarse screen, not an exhaustive list: a molecule that matches is dropped
# entirely, so it is never scored, ranked or given a synthesis procedure.
HAZARD_PATTERNS = {
    "4-anilidopiperidine (opioid-type) scaffold": "[CX3](=O)N(c)[CX4;R]1CCN(CC1)",
    "pethidine-type 4-arylpiperidine-4-carboxylate": "c[C;R]1(C(=O)O[#6])CCN(C)CC1",
    "amphetamine / phenethylamine-type": "c[CH2][CH]([CH3])[NX3;!$(N-C=O);!$(N-S=O)]",
    "cathinone-type": "c[CX3](=O)[CH]([CH3])[NX3;!$(N-C=O);!$(N-S=O)]",
    "tryptamine-type": "c1[nH]c2ccccc2c1CC[NX3;!$(N-C=O);!$(N-S=O)]",
    "barbiturate": "O=C1NC(=O)C([#6])([#6])C(=O)N1",
    "benzodiazepine core": "C1(c)=NCC(=O)Nc2ccccc12",
    "azide": "N=[N+]=[N-]",
    "peroxide": "[OX2][OX2]",
    "N-nitroso": "N-N=O",
    "nitrogen / sulfur mustard": "[N,S](-[CH2][CH2][Cl,Br,I])-[CH2][CH2][Cl,Br,I]",
}
_HAZARD_MOLS = {name: Chem.MolFromSmarts(s) for name, s in HAZARD_PATTERNS.items()}


def hazard_flags(mol: Chem.Mol) -> list[str]:
    flags = [name for name, patt in _HAZARD_MOLS.items() if mol.HasSubstructMatch(patt)]
    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    if len(mol.GetSubstructMatches(nitro)) >= 2:
        flags.append("polynitro (energetic) compound")
    return flags


@dataclass
class Assessment:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    brenk_alerts: int = 0


@lru_cache(maxsize=1)
def _hard_catalog() -> FilterCatalog:
    """PAINS (assay interference) and NIH (reactive / toxic) structural alerts."""
    params = FilterCatalogParams()
    for catalog in (
        FilterCatalogParams.FilterCatalogs.PAINS,
        FilterCatalogParams.FilterCatalogs.NIH,
    ):
        params.AddCatalog(catalog)
    return FilterCatalog(params)


@lru_cache(maxsize=1)
def _soft_catalog() -> FilterCatalog:
    """Brenk unwanted-functionality alerts. Counted as a penalty, not a rejection."""
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(params)


@lru_cache(maxsize=2)
def _single_catalog(kind: str) -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, kind))
    return FilterCatalog(params)


def nih_alerts(mol: Chem.Mol) -> list[str]:
    """NIH reactive / toxic structural alerts only (PAINS excluded)."""
    return [m.GetDescription() for m in _single_catalog("NIH").GetMatches(mol)]


def _lipinski_violations(mol: Chem.Mol) -> int:
    return sum(
        [
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            rdMolDescriptors.CalcNumHBD(mol) > 5,
            rdMolDescriptors.CalcNumHBA(mol) > 10,
        ]
    )


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
    reasons += [f"hazard screen: {flag}" for flag in hazard_flags(mol)]

    mw = Descriptors.MolWt(mol)
    if not MW_RANGE[0] <= mw <= MW_RANGE[1]:
        reasons.append(f"molecular weight {mw:.0f} outside {MW_RANGE}")
    if _lipinski_violations(mol) > MAX_LIPINSKI_VIOLATIONS:
        reasons.append("fails Lipinski rule of five")

    for entry in _hard_catalog().GetMatches(mol):
        reasons.append(f"structural alert: {entry.GetDescription()}")

    return Assessment(
        passed=not reasons,
        reasons=reasons,
        brenk_alerts=len(_soft_catalog().GetMatches(mol)),
    )
