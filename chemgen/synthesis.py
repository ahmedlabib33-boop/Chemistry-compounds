"""One-step template retrosynthesis and a bench-style procedure for each proposed route.

Six common, well-precedented reaction classes are covered (amide coupling, sulfonamide
formation, reductive amination, SNAr amination, Suzuki coupling, aryl-ether alkylation).
Each retro-template splits the target into an electrophile and a nucleophile; routes are
ranked by precursor complexity and by chemoselectivity conflicts (other groups in a
precursor that would compete in the same reaction).

The procedures are generic textbook conditions for the reaction class with quantities
computed for the target. They are not substrate-optimised and not experimentally
validated. Precursor availability is not checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from . import chem, safety

PROTON_MASS = 1.007276

_AMINE = "[NX3;H2,H1;!a;!$(N-[#6,#16]=[O,S,N])]"
_ACID = "[CX3](=O)[OX2H1]"
_PHENOL = "c[OX2H1]"
_ALCOHOL = "[CX4][OX2H1]"
_CARBONYL = "[CX3;!$(C-[#7,#8,#16])]=O"
_ARYL_BR_I = "c[Br,I]"
_ALKYL_HALIDE = "[CX4][Cl,Br,I]"


# Equivalents of (electrophile, nucleophile) per reaction; the limiting reagent is 1.0.
EQUIVALENTS = {"amide": (1.0, 1.1), "sulfonamide": (1.0, 1.1), "reductive_amination": (1.0, 1.1),
               "snar": (1.0, 1.5), "suzuki": (1.0, 1.3), "ether": (1.2, 1.0),
               "ester": (1.0, 1.5), "urea": (1.0, 1.1), "boc": (1.2, 1.0)}


@dataclass(frozen=True)
class Template:
    key: str
    name: str
    retro: str  # product >> electrophile . nucleophile
    bias: float  # small preference for the more dependable reactions
    electrophile_guards: tuple[tuple[str, int], ...]  # (SMARTS, tolerated count)
    nucleophile_guards: tuple[tuple[str, int], ...]


TEMPLATES = (
    Template(
        "amide", "Amide coupling (HATU)",
        "[#6:4][C:1](=[O:2])-!@[NX3;H1,H0;!a;!$(N-S(=O)=O);!$(N(-C=O)-C=O):3]>>[#6:4][C:1](=[O:2])O.[N:3]",
        0.0, ((_ACID, 1), (_AMINE, 0)), ((_AMINE, 1), (_ACID, 0))),
    Template(
        "sulfonamide", "Sulfonamide formation",
        "[#6:4][S:1](=[O:2])(=[O:5])-!@[NX3;H1,H0;!a;!$(N-[#6]=O):3]>>[#6:4][S:1](=[O:2])(=[O:5])Cl.[N:3]",
        0.1, (("S(=O)(=O)Cl", 1), (_AMINE, 0)), ((_AMINE, 1), (_ACID, 0))),
    Template(
        "reductive_amination", "Reductive amination (NaBH(OAc)3)",
        "[CX4;H1,H2;!$(C(-[N,O,S])-[N,O,S]):1]-!@[NX3;H1,H0;!a;!$(N-[#6,#16]=[O,S,N]):2]>>[C:1]=O.[N:2]",
        0.3, ((_CARBONYL, 1), (_AMINE, 0)), ((_AMINE, 1),)),
    Template(
        "snar", "SNAr amination",
        "[c;$(c:n):1]-!@[NX3;H1,H0;!a;!$(N-[#6,#16]=[O,S,N]):2]>>Cl[c:1].[N:2]",
        0.5, (("[c;$(c:n)][F,Cl,Br]", 1),), ((_AMINE, 1),)),
    Template(
        "suzuki", "Suzuki-Miyaura coupling",
        "[c:1]-!@[c:2]>>Br[c:1].OB(O)[c:2]",
        0.8, ((_ARYL_BR_I, 1),), (("cB(O)O", 1), (_ARYL_BR_I, 0))),
    Template(
        "ether", "Aryl-ether alkylation (Williamson)",
        "[CX4;H2:1]-!@[OX2:2]-!@[c:3]>>Br[C:1].[O:2][c:3]",
        0.5, ((_ALKYL_HALIDE, 1), (_AMINE, 0), (_PHENOL, 0), (_ACID, 0)),
        ((_PHENOL, 1), (_AMINE, 0), (_ACID, 0))),
    Template(
        "ester", "Ester coupling (EDC/DMAP)",
        "[#6:4][C:1](=[O:2])-!@[OX2:3]-[CX4;!H0:5]>>[#6:4][C:1](=[O:2])O.[OX2:3][C:5]",
        0.4, ((_ACID, 1), (_AMINE, 0)), ((_ALCOHOL, 1), (_AMINE, 0), (_ACID, 0))),
    Template(
        "urea", "Urea formation (CDI)",
        "[NX3;H1,H0;!a:1]-!@[CX3](=O)-!@[NX3;H1,H0;!a:2]>>[N:1].[N:2]",
        0.4, ((_AMINE, 1), (_ACID, 0)), ((_AMINE, 1), (_ACID, 0))),
    Template(
        "boc", "Boc protection (Boc2O)",
        "[NX3;H1,H0;!a:1]-!@[CX3](=O)[OX2][CX4]([CH3])([CH3])[CH3]>>CC(C)(C)OC(=O)OC(=O)OC(C)(C)C.[N:1]",
        0.2, (), ((_AMINE, 1),)),
)
_BY_KEY = {t.key: t for t in TEMPLATES}


@lru_cache(maxsize=None)
def _reaction(key: str):
    return AllChem.ReactionFromSmarts(_BY_KEY[key].retro)


@dataclass
class Route:
    template: Template
    electrophile: Chem.Mol
    nucleophile: Chem.Mol
    conflicts: int
    cost: float


def _conflicts(mol: Chem.Mol, guards: tuple[tuple[str, int], ...]) -> int:
    total = 0
    for smarts, tolerated in guards:
        total += max(0, len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))) - tolerated)
    return total


def _prepare(mol: Chem.Mol) -> Chem.Mol:
    """Drop explicit H counts (e.g. from [C@H]) so atoms can change valence in a template.

    Implicit Hs encode the same stereo parity, so stereochemistry is unaffected.
    """
    prepared = Chem.RWMol(mol)
    for atom in prepared.GetAtoms():
        if not atom.GetIsAromatic() and atom.GetFormalCharge() == 0:
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
    Chem.SanitizeMol(prepared)
    return prepared.GetMol()


def _disconnections(mol: Chem.Mol, template: Template):
    seen, found = set(), []
    for products in _reaction(template.key).RunReactants((_prepare(mol),)):
        try:
            for p in products:
                Chem.SanitizeMol(p)
            if any(len(Chem.GetMolFrags(p)) != 1 for p in products):
                continue
            key = tuple(Chem.MolToSmiles(p) for p in products)
        except Exception:
            continue
        if key not in seen:
            seen.add(key)
            found.append(products)
    return found


def propose_routes(mol: Chem.Mol) -> list[Route]:
    """All one-step disconnections, cheapest first."""
    routes = []
    for template in TEMPLATES:
        for electrophile, nucleophile in _disconnections(mol, template):
            # Never write up a route whose building block is itself a screened structure.
            if safety.hazard_flags(electrophile) or safety.hazard_flags(nucleophile):
                continue
            conflicts = (_conflicts(electrophile, template.electrophile_guards)
                         + _conflicts(nucleophile, template.nucleophile_guards))
            cost = (max(chem.sa_score(electrophile), chem.sa_score(nucleophile))
                    + 2.0 * conflicts
                    + 0.05 * max(electrophile.GetNumHeavyAtoms(), nucleophile.GetNumHeavyAtoms())
                    + template.bias)
            routes.append(Route(template, electrophile, nucleophile, conflicts, cost))
    return sorted(routes, key=lambda r: r.cost)


@dataclass(frozen=True)
class Reagent:
    name: str
    role: str
    equiv: float
    mw: float
    density: float | None = None  # g/mL, for liquids
    amount_text: str | None = None  # overrides the computed amount (e.g. aqueous solutions)

    def amount(self, scale_mmol: float) -> str:
        if self.amount_text:
            return self.amount_text
        mmol = self.equiv * scale_mmol
        mg = mmol * self.mw
        if self.density:
            return f"{mmol:.2f} mmol, {mg / self.density:.0f} µL"
        return f"{mmol:.2f} mmol, {mg:.0f} mg"


@dataclass
class Recipe:
    target_smiles: str
    formula: str
    mw: float
    exact_mass: float
    reaction_name: str
    template_key: str
    electrophile_smiles: str
    nucleophile_smiles: str
    scale_mmol: float
    reagents: list[tuple[str, str, str, str]]  # name, role, equiv, amount
    solvent: str
    temperature: str
    duration: str
    steps: list[str]
    workup: list[str]
    purification: str
    hazards: list[str]
    warnings: list[str] = field(default_factory=list)
    alternatives: int = 0
    electrophile_mmol: float = 0.0  # consumed by this step, for scaling the step that makes it
    nucleophile_mmol: float = 0.0

    @property
    def summary(self) -> str:
        return f"{self.reaction_name}: {self.electrophile_smiles} + {self.nucleophile_smiles}"

    def to_markdown(self, title: str | None = None) -> str:
        theo = self.mw * self.scale_mmol
        lines = [
            title or f"### {self.reaction_name}: one-step route",
            f"**Target:** `{self.target_smiles}`  ({self.formula}, MW {self.mw:.2f}, "
            f"exact mass {self.exact_mass:.4f}, expected [M+H]+ {self.exact_mass + PROTON_MASS:.4f})",
            "",
            f"**Scheme:** `{self.electrophile_smiles}` + `{self.nucleophile_smiles}` → `{self.target_smiles}`",
            "",
            f"**Reagents** ({self.scale_mmol:g} mmol scale; theoretical yield "
            f"{theo:.0f} mg at 100 %)",
            "",
            "| Reagent | Role | Equiv | Amount |",
            "| --- | --- | --- | --- |",
            *[f"| {n} | {r} | {e} | {a} |" for n, r, e, a in self.reagents],
            "",
            f"**Solvent:** {self.solvent}",
            f"**Temperature / time:** {self.temperature}; {self.duration}",
            "",
            "**Procedure**",
            *[f"{i}. {s}" for i, s in enumerate(self.steps, 1)],
            "",
            "**Work-up**",
            *[f"{i}. {s}" for i, s in enumerate(self.workup, 1)],
            "",
            f"**Purification:** {self.purification}",
            "",
            "**Characterisation:** 1H and 13C NMR; HRMS ([M+H]+ above); HPLC purity > 95 % "
            "before any use beyond chemistry.",
            "",
            "**Hazards and handling**",
            *[f"- {h}" for h in self.hazards],
        ]
        if self.warnings:
            lines += ["", "**Route warnings**", *[f"- {w}" for w in self.warnings]]
        lines += [
            "",
            f"*{self.alternatives} other disconnection(s) were found and ranked lower.*",
            "",
            "> Computer-generated proposal. Conditions are generic for this reaction class, "
            "not optimised for this substrate, and no yield is predicted. Precursor "
            "availability was not checked. Confirm literature precedent (SciFinder/Reaxys), "
            "carry out a formal risk assessment, and work only in a properly equipped "
            "laboratory under the supervision of a trained chemist. The target is a novel "
            "structure of unknown toxicity: treat it as hazardous.",
        ]
        return "\n".join(lines)


_GENERAL_HAZARDS = [
    "Work in a fume hood with lab coat, nitrile gloves and eye protection.",
    "The product and intermediates are uncharacterised compounds of unknown toxicity.",
]


def _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, e_role, n_role, n_equiv, scale):
    e = Reagent(f"`{e_smiles}`", e_role, 1.0, e_mw)
    n = Reagent(f"`{n_smiles}`", n_role, n_equiv, n_mw)
    return [e, n]


def _rows(reagents: list[Reagent], scale: float) -> list[tuple[str, str, str, str]]:
    return [(r.name, r.role, f"{r.equiv:g}", r.amount(scale)) for r in reagents]


def _procedure(key, e_smiles, n_smiles, e_mw, n_mw, scale):
    """Return (reagents, solvent, temperature, duration, steps, workup, purification, hazards)."""
    e_amt = f"{scale * e_mw:.0f} mg"
    if key == "amide":
        n_eq = EQUIVALENTS["amide"][1]
        reagents = _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, "carboxylic acid (limiting)",
                                   "amine", n_eq, scale) + [
            Reagent("HATU", "coupling agent", 1.2, 380.23),
            Reagent("DIPEA", "base", 3.0, 129.25, density=0.742)]
        steps = [
            "Oven-dry a round-bottom flask with a stir bar and flush with nitrogen or argon.",
            f"Add the carboxylic acid ({e_amt}, {scale:g} mmol) and anhydrous DMF "
            f"({10 * scale:g} mL, 0.1 M).",
            "Cool to 0 °C. Add DIPEA (3.0 equiv), then HATU (1.2 equiv); stir 10-15 min to "
            "pre-activate the acid.",
            f"Add the amine ({n_eq} equiv). If it is a hydrochloride salt, add a further 1.0 equiv of DIPEA.",
            "Remove the ice bath and stir at room temperature. Follow by LC-MS or TLC every "
            "~2 h until the acid is consumed.",
        ]
        workup = [
            f"Dilute with ethyl acetate ({50 * scale:g} mL).",
            f"Wash with saturated aqueous NaHCO3 (2 × {20 * scale:g} mL), 5 % aqueous LiCl "
            f"(3 × {20 * scale:g} mL, to remove DMF) and brine ({20 * scale:g} mL). Skip acid "
            "washes if the product is basic.",
            "Dry over Na2SO4, filter and concentrate under reduced pressure.",
        ]
        return (reagents, "DMF, anhydrous (0.1 M)", "0 °C, then room temperature", "2-16 h",
                steps, workup,
                "Flash chromatography on silica (hexanes/ethyl acetate or DCM/methanol gradient) "
                "or reverse-phase HPLC.",
                ["HATU is a skin and respiratory sensitiser: avoid dust and skin contact.",
                 "DMF is a reproductive toxicant and is absorbed through skin.",
                 "DIPEA is flammable and corrosive."])
    if key == "sulfonamide":
        n_eq = EQUIVALENTS["sulfonamide"][1]
        reagents = _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, "sulfonyl chloride (limiting)",
                                   "amine", n_eq, scale) + [
            Reagent("Triethylamine", "base", 2.0, 101.19, density=0.726)]
        steps = [
            "Oven-dry a flask with a stir bar and flush with nitrogen.",
            f"Dissolve the amine ({n_eq} equiv) and triethylamine (2.0 equiv) in anhydrous "
            f"dichloromethane ({5 * scale:g} mL) and cool to 0 °C.",
            f"Add the sulfonyl chloride ({e_amt}, {scale:g} mmol) in portions or as a solution in "
            f"DCM ({2 * scale:g} mL) over 5 min.",
            "Stir at 0 °C for 30 min, then at room temperature; follow by TLC or LC-MS.",
        ]
        workup = [
            f"Dilute with dichloromethane ({20 * scale:g} mL).",
            f"Wash with 1 M HCl ({15 * scale:g} mL; omit if the product is basic), saturated "
            f"aqueous NaHCO3 ({15 * scale:g} mL) and brine.",
            "Dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "Dichloromethane, anhydrous (0.2 M)", "0 °C, then room temperature",
                "2-12 h", steps, workup,
                "Recrystallisation or flash chromatography (hexanes/ethyl acetate).",
                ["Sulfonyl chlorides are corrosive, lachrymatory and moisture-sensitive; they "
                 "release HCl on contact with water.",
                 "Dichloromethane is volatile and a suspected carcinogen."])
    if key == "reductive_amination":
        n_eq = EQUIVALENTS["reductive_amination"][1]
        reagents = _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, "aldehyde/ketone (limiting)",
                                   "amine", n_eq, scale) + [
            Reagent("Acetic acid", "catalyst", 1.0, 60.05, density=1.049),
            Reagent("Sodium triacetoxyborohydride", "reductant", 1.5, 211.94)]
        steps = [
            f"In a dry flask under nitrogen, dissolve the carbonyl compound ({e_amt}, "
            f"{scale:g} mmol) and the amine ({n_eq} equiv) in dichloromethane ({10 * scale:g} mL).",
            "Add acetic acid (1.0 equiv) and stir 30 min at room temperature to form the "
            "imine/iminium.",
            "Add sodium triacetoxyborohydride (1.5 equiv) portionwise over 10 min (mild gas evolution).",
            "Stir at room temperature; follow by LC-MS until the imine and carbonyl are consumed.",
        ]
        workup = [
            f"Quench slowly with saturated aqueous NaHCO3 ({10 * scale:g} mL) (gas evolution) and "
            "stir 15 min.",
            f"Extract with dichloromethane (3 × {15 * scale:g} mL); wash the combined organics with brine.",
            "Dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "Dichloromethane, anhydrous (0.1 M; THF also works)", "Room temperature",
                "4-16 h", steps, workup,
                "Flash chromatography; for basic amine products add 1 % triethylamine or "
                "NH3/methanol to the eluent.",
                ["NaBH(OAc)3 reacts with water and acids to release hydrogen: add it in portions "
                 "and quench slowly.",
                 "Aldehydes can be irritants or sensitisers.",
                 "Dichloromethane is volatile and a suspected carcinogen."])
    if key == "snar":
        n_eq = EQUIVALENTS["snar"][1]
        reagents = _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, "heteroaryl chloride (limiting)",
                                   "amine", n_eq, scale) + [
            Reagent("DIPEA", "base", 2.5, 129.25, density=0.742)]
        steps = [
            f"In a pressure-rated vial, combine the heteroaryl chloride ({e_amt}, {scale:g} mmol), "
            f"the amine ({n_eq} equiv) and DIPEA (2.5 equiv) in NMP or DMSO ({2 * scale:g} mL).",
            "Seal the vial and heat behind a blast shield (a microwave reactor is also suitable).",
            "Follow by LC-MS every 2-4 h. If conversion stalls, raise the temperature in 20 °C steps.",
        ]
        workup = [
            "Cool to room temperature and dilute with water "
            f"({30 * scale:g} mL). If the product precipitates, filter and wash with water.",
            f"Otherwise extract with ethyl acetate (3 × {20 * scale:g} mL); wash the organics with water "
            "(3×) and brine to remove NMP/DMSO.",
            "Dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "NMP or DMSO (0.5 M)", "100-140 °C (sealed vial)", "4-24 h", steps, workup,
                "Flash chromatography or reverse-phase HPLC.",
                ["Heating a sealed vial builds pressure: use a rated vial and a blast shield.",
                 "NMP is a reproductive toxicant; DMSO carries dissolved compounds through skin.",
                 "Reactivity depends on ring activation; unactivated rings may need Pd catalysis instead."])
    if key == "suzuki":
        n_eq = EQUIVALENTS["suzuki"][1]
        reagents = _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, "aryl bromide (limiting)",
                                   "boronic acid", n_eq, scale) + [
            Reagent("Pd(dppf)Cl2·CH2Cl2", "catalyst", 0.05, 816.64),
            Reagent("Potassium carbonate (2 M aqueous)", "base", 3.0, 138.21,
                    amount_text=f"3.00 mmol, {1.5 * scale:g} mL of 2 M solution")]
        steps = [
            f"To a reaction vial add the aryl bromide ({e_amt}, {scale:g} mmol), the boronic acid "
            f"({n_eq} equiv) and Pd(dppf)Cl2·CH2Cl2 (5 mol %).",
            "Evacuate and backfill with nitrogen three times.",
            f"Add degassed 1,4-dioxane ({8 * scale:g} mL) and degassed 2 M aqueous K2CO3 "
            f"({1.5 * scale:g} mL) (sparge each with nitrogen for 15 min beforehand).",
            "Heat with vigorous stirring; follow by LC-MS. Typical completion 2-16 h.",
        ]
        workup = [
            "Cool to room temperature; dilute with ethyl acetate and filter through Celite, "
            "rinsing with ethyl acetate.",
            "Wash the filtrate with water and brine; dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "1,4-Dioxane / water, about 5:1 (0.1 M)", "80-90 °C", "2-16 h",
                steps, workup,
                "Flash chromatography. For biological work, remove residual palladium (thiol-silica "
                "scavenger) and confirm by ICP-MS.",
                ["Palladium compounds are toxic; collect all Pd-containing waste separately.",
                 "1,4-Dioxane is a suspected carcinogen and forms peroxides: use fresh, inhibited solvent.",
                 "Boronic acids can protodeboronate; use fresh material."])
    if key == "ether":
        e_eq = EQUIVALENTS["ether"][0]
        reagents = [Reagent(f"`{n_smiles}`", "phenol (limiting)", 1.0, n_mw),
                    Reagent(f"`{e_smiles}`", "alkyl bromide", e_eq, e_mw),
                    Reagent("Potassium carbonate", "base", 2.0, 138.21)]
        steps = [
            f"In a dry flask, dissolve the phenol ({scale * n_mw:.0f} mg, {scale:g} mmol) in "
            f"anhydrous DMF ({5 * scale:g} mL) under nitrogen.",
            "Add finely ground K2CO3 (2.0 equiv) and stir 15 min at room temperature.",
            f"Add the alkyl bromide ({e_eq} equiv) and heat to 60 °C; follow by TLC or LC-MS.",
        ]
        workup = [
            f"Cool, dilute with water ({25 * scale:g} mL) and extract with ethyl acetate "
            f"(3 × {15 * scale:g} mL).",
            "Wash the organics with 1 M NaOH (removes unreacted phenol; skip if the product is "
            "base-sensitive), water (3×) and brine.",
            "Dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "DMF, anhydrous (0.2 M; acetonitrile also works)", "60 °C", "4-16 h",
                steps, workup, "Flash chromatography (hexanes/ethyl acetate).",
                ["Alkyl bromides are alkylating agents and may be lachrymators or mutagens: "
                 "handle only in a fume hood.",
                 "DMF is a reproductive toxicant and is absorbed through skin."])
    if key == "ester":
        n_eq = EQUIVALENTS["ester"][1]
        reagents = _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, "carboxylic acid (limiting)",
                                   "alcohol", n_eq, scale) + [
            Reagent("EDC·HCl", "coupling agent", 1.5, 191.70),
            Reagent("DMAP", "catalyst", 0.1, 122.17)]
        steps = [
            "Oven-dry a flask with a stir bar and flush with nitrogen.",
            f"Dissolve the carboxylic acid ({e_amt}, {scale:g} mmol), the alcohol ({n_eq} equiv) and DMAP "
            f"(0.1 equiv) in anhydrous dichloromethane ({10 * scale:g} mL) and cool to 0 °C.",
            "Add EDC·HCl (1.5 equiv) in one portion.",
            "Allow to warm to room temperature and stir; follow by TLC or LC-MS until the acid is consumed.",
        ]
        workup = [
            f"Dilute with dichloromethane ({20 * scale:g} mL).",
            f"Wash with 1 M HCl ({15 * scale:g} mL; omit if the product is basic), saturated aqueous NaHCO3 "
            f"({15 * scale:g} mL) and brine; this removes the urea by-product, DMAP and unreacted acid.",
            "Dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "Dichloromethane, anhydrous (0.1 M)", "0 °C, then room temperature", "4-16 h",
                steps, workup, "Flash chromatography (hexanes/ethyl acetate).",
                ["EDC·HCl is a skin and respiratory sensitiser.",
                 "DMAP is toxic and readily absorbed through the skin: wear double gloves.",
                 "Dichloromethane is volatile and a suspected carcinogen."])
    if key == "urea":
        n_eq = EQUIVALENTS["urea"][1]
        reagents = _substrate_rows(e_smiles, n_smiles, e_mw, n_mw, "first amine (limiting)",
                                   "second amine", n_eq, scale) + [
            Reagent("1,1'-Carbonyldiimidazole (CDI)", "carbonyl source", 1.1, 162.15)]
        steps = [
            "Oven-dry a flask with a stir bar and flush with nitrogen.",
            f"Dissolve CDI (1.1 equiv) in anhydrous THF ({5 * scale:g} mL) and add the first amine "
            f"({e_amt}, {scale:g} mmol) dropwise as a solution in THF ({5 * scale:g} mL).",
            "Stir 1-2 h at room temperature until LC-MS shows the carbamoyl-imidazole intermediate "
            "(gentle CO2 evolution; keep the vessel vented through a bubbler).",
            f"Add the second amine ({n_eq} equiv). If it is a hydrochloride salt, add 1.0 equiv of triethylamine.",
            "Stir at room temperature, or warm to 50 °C if conversion is slow; follow by LC-MS.",
        ]
        workup = [
            f"Concentrate, redissolve in ethyl acetate ({30 * scale:g} mL).",
            f"Wash with water (2 × {15 * scale:g} mL) to remove imidazole, then 0.5 M HCl (omit if the product "
            "is basic) and brine.",
            "Dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "THF, anhydrous (0.1 M)", "Room temperature (up to 50 °C)", "4-16 h", steps, workup,
                "Recrystallisation or flash chromatography (DCM/methanol).",
                ["CDI is moisture-sensitive and a sensitiser; it releases CO2 and imidazole.",
                 "Add the first amine slowly to avoid symmetrical urea by-product."])
    if key == "boc":
        e_eq = EQUIVALENTS["boc"][0]
        reagents = [Reagent(f"`{n_smiles}`", "amine (limiting)", 1.0, n_mw),
                    Reagent("Di-tert-butyl dicarbonate (Boc2O)", "protecting reagent", e_eq, 218.25, density=0.95),
                    Reagent("Triethylamine", "base", 1.5, 101.19, density=0.726)]
        steps = [
            f"Dissolve the amine ({scale * n_mw:.0f} mg, {scale:g} mmol) in dichloromethane ({5 * scale:g} mL) "
            "and add triethylamine (1.5 equiv); if the amine is a hydrochloride salt add a further 1.0 equiv.",
            "Cool to 0 °C and add Boc2O (1.2 equiv) neat or as a solution in dichloromethane.",
            "Allow to warm to room temperature and stir with a vent to a bubbler (CO2 is released); follow by TLC "
            "(ninhydrin stain loses the amine spot) or LC-MS.",
        ]
        workup = [
            f"Dilute with dichloromethane ({15 * scale:g} mL) and wash with 10 % aqueous citric acid (omit if the "
            "product is basic), saturated aqueous NaHCO3 and brine.",
            "Dry over Na2SO4, filter and concentrate.",
        ]
        return (reagents, "Dichloromethane (0.2 M)", "0 °C, then room temperature", "2-12 h", steps, workup,
                "Flash chromatography (hexanes/ethyl acetate).",
                ["Boc2O is toxic if inhaled, flammable and a sensitiser: handle only in a fume hood.",
                 "Never seal the reaction: CO2 is released and pressure builds.",
                 "Dichloromethane is volatile and a suspected carcinogen."])
    raise KeyError(key)


def build_recipe(target: Chem.Mol, route: Route, scale_mmol: float = 1.0,
                 alternatives: int = 0) -> Recipe:
    e_smiles = Chem.MolToSmiles(route.electrophile)
    n_smiles = Chem.MolToSmiles(route.nucleophile)
    (reagents, solvent, temperature, duration, steps, workup, purification,
     hazards) = _procedure(route.template.key, e_smiles, n_smiles,
                           Descriptors.MolWt(route.electrophile),
                           Descriptors.MolWt(route.nucleophile), scale_mmol)

    warnings = []
    if route.conflicts:
        warnings.append(
            f"{route.conflicts} competing reactive group(s) in the precursors may react in "
            "the same step. Protect them (e.g. Boc on amines, ester on acids) or reorder the "
            "synthesis.")
    n_stereo = len(Chem.FindMolChiralCenters(target, includeUnassigned=False, useLegacyImplementation=False))
    if n_stereo:
        warnings.append(
            f"The target specifies {n_stereo} stereocentre(s). Use enantiopure precursors, confirm "
            "that their configuration matches, and check ee/dr by chiral HPLC.")
    if route.template.key == "reductive_amination":
        warnings.append("Reductive amination of a ketone creates a new stereocentre as a mixture "
                        "unless a chiral amine or catalyst controls it.")
    if max(chem.sa_score(route.electrophile), chem.sa_score(route.nucleophile)) > 3.5:
        warnings.append("At least one precursor is itself fairly complex and may need its own synthesis.")

    return Recipe(
        target_smiles=Chem.MolToSmiles(target),
        formula=rdMolDescriptors.CalcMolFormula(target),
        mw=Descriptors.MolWt(target),
        exact_mass=Descriptors.ExactMolWt(target),
        reaction_name=route.template.name,
        template_key=route.template.key,
        electrophile_smiles=e_smiles,
        nucleophile_smiles=n_smiles,
        scale_mmol=scale_mmol,
        reagents=_rows(reagents, scale_mmol),
        solvent=solvent, temperature=temperature, duration=duration,
        steps=steps, workup=workup, purification=purification,
        hazards=hazards + _GENERAL_HAZARDS, warnings=warnings, alternatives=alternatives,
        electrophile_mmol=EQUIVALENTS[route.template.key][0] * scale_mmol,
        nucleophile_mmol=EQUIVALENTS[route.template.key][1] * scale_mmol,
    )


def propose_recipe(target: Chem.Mol, scale_mmol: float = 1.0) -> Recipe | None:
    """Best one-step route as a full procedure, or None if no template applies."""
    routes = propose_routes(target)
    if not routes:
        return None
    return build_recipe(target, routes[0], scale_mmol, alternatives=len(routes) - 1)
