"""Automotive additive "role" tagging: a small, honest substitute for the activity data that does
not exist in open, downloadable form for corrosion inhibition, antiwear/EP performance, friction
modification or antioxidant performance.

Each tag is a **rule-based functional-group match** (SMARTS, classic motifs from the lubricant
and fuel-additive literature) -- never a measured or learned activity, and never a potency score.
A match says "this structural class is traditionally used this way", nothing about how well a
specific molecule would perform; most matches below are deliberately broad, so expect some
false positives. Report every matched tag; do not pick just one.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


@dataclass(frozen=True)
class Role:
    key: str
    label: str
    note: str
    smarts: tuple[str, ...]           # any one match is enough
    extra: callable = None            # optional additional structural test on the whole molecule


def _is_mostly_hydrocarbon(mol: Chem.Mol, max_heteroatoms: int = 1) -> bool:
    hetero = sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in ("C", "H"))
    return hetero <= max_heteroatoms


def _long_chain(mol: Chem.Mol, min_carbons: int = 8) -> bool:
    return len(mol.GetSubstructMatches(Chem.MolFromSmarts("[CH2]" * min_carbons))) > 0


ROLES = (
    Role("corrosion_inhibitor", "Corrosion inhibitor",
        "Film-forming / metal-adsorbing motif (azole, mercaptothiazole, imidazoline, or a long-chain "
        "fatty amine) seen in oilfield, coolant and metalworking-fluid inhibitors.",
        ("c1ccc2[nH]nnc2c1", "c1ccc2nc([SH1,S-])sc2c1", "C1CN=C(N1)[CH2][CH2][CH2][CH2]",
         "[NX3;H2][CH2][CH2][CH2][CH2][CH2][CH2]")),
    Role("antiwear_ep", "Antiwear / extreme-pressure additive",
        "Organophosphorus ester or organic di-/polysulfide, the backbone chemistry of most "
        "antiwear (AW) and extreme-pressure (EP) lubricant additives (e.g. TCP-, ZDDP- and "
        "sulfurised-olefin-type).",
        ("[#6]O[PX4](=[O,S])(O[#6])O[#6]", "[#6]O[PX4](=[O,S])(O[#6])[#6]",
         "[#6][SX2][SX2][#6]", "[#6]O[BX3](O[#6])O[#6]")),
    Role("antioxidant", "Antioxidant",
        "Hindered phenol or diarylamine: the two dominant antioxidant chemotypes in fuels and "
        "lubricants, which scavenge the radicals that drive oxidative degradation.",
        ("c1(O)c(C(C)(C)C)cc([#6,#1])cc1C(C)(C)C", "c1ccccc1Nc1ccccc1")),
    Role("friction_modifier", "Organic friction modifier",
        "Long-chain fatty ester or fatty amide with a polar head group, the classic organic "
        "friction-modifier pattern (adsorbs on metal to cut boundary-lubrication friction).",
        ("[CH2][CH2][CH2][CH2][CH2][CH2]C(=O)[OX2H0,NX3]",)),
    Role("base_oil_like", "Base-oil-like hydrocarbon",
        "Mostly-aliphatic hydrocarbon in roughly the base-oil molecular-weight range "
        "(no performance claim -- lubricant base oils are complex mixtures, not single molecules).",
        (), extra=lambda m: _is_mostly_hydrocarbon(m) and 220 <= Descriptors.MolWt(m) <= 600
                            and rdMolDescriptors.CalcNumRotatableBonds(m) >= 6),
    Role("fuel_range_hydrocarbon", "Fuel-range hydrocarbon",
        "Hydrocarbon (or hydrocarbon + ether/ester oxygenate) in roughly the gasoline/diesel "
        "molecular-weight range.",
        (), extra=lambda m: _is_mostly_hydrocarbon(m, max_heteroatoms=2) and 55 <= Descriptors.MolWt(m) <= 280
                            and rdMolDescriptors.CalcTPSA(m) <= 30),
)
_COMPILED = {r.key: tuple(Chem.MolFromSmarts(s) for s in r.smarts) for r in ROLES}


@dataclass(frozen=True)
class RoleMatch:
    key: str
    label: str
    note: str


def classify(mol: Chem.Mol) -> list[RoleMatch]:
    matches = []
    for role in ROLES:
        hit = any(mol.HasSubstructMatch(p) for p in _COMPILED[role.key])
        if not hit and role.extra is not None:
            hit = bool(role.extra(mol))
        if hit:
            matches.append(RoleMatch(role.key, role.label, role.note))
    return matches
