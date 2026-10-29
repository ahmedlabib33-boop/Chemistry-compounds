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
