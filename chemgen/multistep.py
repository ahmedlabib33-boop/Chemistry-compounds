"""Multi-step route planning: chain the one-step reaction classes until every branch ends in a
simple starting material, then write a detailed procedure for each step with quantities scaled
back from the final step.

Search: a molecule that is not a simple building block is disconnected with the best few
one-step routes, and each precursor is planned recursively within the remaining step budget. The
cheapest plan wins (one unit per step, extra for chemoselectivity conflicts, a large penalty for
a complex precursor that cannot be broken down). Every intermediate and starting material goes
through the hazard screen inside ``synthesis.propose_routes``.

"Simple building block" is a size/complexity heuristic, not a catalogue lookup. Use
``MultiStepRecipe.annotate`` to check starting materials against PubChem (a known compound is
not necessarily a purchasable one).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from rdkit import Chem
from rdkit.Chem import Descriptors

from . import chem, synthesis

STEP_YIELD = 0.8          # planning assumption used to scale earlier steps, not a prediction
UNRESOLVED_PENALTY = 6.0
TOP_ALTERNATIVES = 3
MAX_STEPS_LIMIT = 6


def is_building_block(mol: Chem.Mol) -> bool:
    heavy = mol.GetNumHeavyAtoms()
    return heavy <= 12 or (heavy <= 18 and chem.sa_score(mol) <= 2.5)


@dataclass
class PlanNode:
    mol: Chem.Mol
    smiles: str
    route: synthesis.Route | None = None           # None: a starting material
    children: list["PlanNode"] = field(default_factory=list)   # [electrophile, nucleophile]
    unresolved: bool = False                       # complex, but no further disconnection found

    @property
    def steps(self) -> int:
        return 0 if self.route is None else 1 + sum(c.steps for c in self.children)

    @property
    def depth(self) -> int:
        return 0 if self.route is None else 1 + max(c.depth for c in self.children)


def _leaf(mol: Chem.Mol, unresolved: bool = False) -> PlanNode:
    return PlanNode(mol, Chem.MolToSmiles(mol), unresolved=unresolved)


def _solve(mol: Chem.Mol, budget: int, root: bool, memo: dict) -> tuple[float, PlanNode]:
    smiles = Chem.MolToSmiles(mol)
    key = (smiles, budget, root)
    if key in memo:
        return memo[key]
    if not root and is_building_block(mol):
        result = (0.0, _leaf(mol))
    elif budget <= 0:
        result = (UNRESOLVED_PENALTY, _leaf(mol, unresolved=True))
    else:
        best = None
        for route in synthesis.propose_routes(mol)[:TOP_ALTERNATIVES]:
            cost_e, node_e = _solve(route.electrophile, budget - 1, False, memo)
            cost_n, node_n = _solve(route.nucleophile, budget - 1 - node_e.steps, False, memo)
            total = 1 + 2.0 * route.conflicts + cost_e + cost_n
            if best is None or total < best[0]:
                best = (total, PlanNode(mol, smiles, route, [node_e, node_n]))
        result = best or (UNRESOLVED_PENALTY, _leaf(mol, unresolved=True))
    memo[key] = result
    return result


@dataclass
class StepInput:
    smiles: str
    mmol: float
    from_step: int | None      # None: a starting material
    unresolved: bool = False


@dataclass
class PlanStep:
    number: int
    product: str
    recipe: synthesis.Recipe
    inputs: list[StepInput]


@dataclass
class StartingMaterial:
    smiles: str
    mmol: float
    mw: float
    unresolved: bool = False
    pubchem_cid: int | None = None
    checked: bool = False

    @property
    def mg(self) -> float:
        return self.mmol * self.mw


@dataclass
class MultiStepRecipe:
    target_smiles: str
    formula: str
    mw: float
    exact_mass: float
    scale_mmol: float
    steps: list[PlanStep]
    starting_materials: list[StartingMaterial]
    longest_sequence: int
    warnings: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def summary(self) -> str:
        names = " → ".join(s.recipe.reaction_name for s in self.steps)
        plural = "step" if self.n_steps == 1 else "steps"
        return f"{self.n_steps} {plural}: {names}"

    def annotate(self, lookup: Callable[[str], int | None]) -> None:
        """Record whether each starting material is a known PubChem compound."""
        for material in self.starting_materials:
            try:
                material.pubchem_cid, material.checked = lookup(material.smiles), True
            except Exception:  # offline or rate-limited: leave it marked as not checked
                continue

    def to_markdown(self) -> str:
        total = self.n_steps
        lines = [
            f"# Preparing `{self.target_smiles}` in {total} step{'s' if total != 1 else ''}",
            f"**Target:** {self.formula}, MW {self.mw:.2f}, expected [M+H]+ "
            f"{self.exact_mass + synthesis.PROTON_MASS:.4f}  ·  longest linear sequence "
            f"{self.longest_sequence} step{'s' if self.longest_sequence != 1 else ''}  ·  final scale "
            f"{self.scale_mmol:g} mmol",
            "",
            "## Route overview",
        ]
        for step in self.steps:
            made_from = " + ".join(
                f"`{i.smiles}`" + (f" (from step {i.from_step})" if i.from_step else "") for i in step.inputs)
            lines.append(f"{step.number}. {made_from} → `{step.product}`  ({step.recipe.reaction_name})")
        overall = STEP_YIELD ** self.longest_sequence
        lines += [
            "",
            f"*Planning assumption, not a prediction:* each step is taken to give {STEP_YIELD:.0%} of "
            f"theoretical yield, so quantities for earlier steps are scaled up by 1/{STEP_YIELD:g}; over the "
            f"longest sequence that would deliver about {overall:.0%} of the final theoretical amount.",
            "",
            "## Starting materials",
            "| Compound | Needed | Mass | Status |",
            "| --- | --- | --- | --- |",
        ]
        for m in self.starting_materials:
            if m.unresolved:
                status = "complex precursor with no further route found: source or prepare it separately"
            elif not m.checked:
                status = "simple building block (availability not checked)"
            elif m.pubchem_cid:
                status = f"known compound, PubChem CID {m.pubchem_cid} (check a supplier)"
            else:
                status = "not found in PubChem: confirm it can be sourced or made"
            lines.append(f"| `{m.smiles}` | {m.mmol:.2f} mmol | {m.mg:.0f} mg | {status} |")
        for step in self.steps:
            lines += ["", "---", "", f"**Inputs for step {step.number}**"]
            for i in step.inputs:
                where = f"product of step {i.from_step}" if i.from_step else "starting material"
                lines.append(f"- `{i.smiles}`: {i.mmol:.2f} mmol, {where}")
            lines += ["", step.recipe.to_markdown(title=f"## Step {step.number} of {total}: {step.recipe.reaction_name}")]
        if self.warnings:
            lines += ["", "---", "", "**Route warnings**", *[f"- {w}" for w in self.warnings]]
        return "\n".join(lines)


def propose_plan(target: Chem.Mol, scale_mmol: float = 1.0, max_steps: int = 3) -> MultiStepRecipe | None:
    """Best route of at most ``max_steps`` reactions, or None if no reaction class applies."""
    max_steps = max(1, min(max_steps, MAX_STEPS_LIMIT))
    _, root = _solve(target, max_steps, True, {})
    if root.route is None:
        return None

    steps: list[PlanStep] = []
    materials: dict[str, StartingMaterial] = {}
    warnings: list[str] = []

    def visit(node: PlanNode, scale: float) -> int:
        recipe = synthesis.build_recipe(node.mol, node.route, scale)
        inputs = []
        for child, need in zip(node.children, (recipe.electrophile_mmol, recipe.nucleophile_mmol)):
            if child.route is None:
                inputs.append(StepInput(child.smiles, need, None, child.unresolved))
                material = materials.setdefault(
                    child.smiles, StartingMaterial(child.smiles, 0.0, Descriptors.MolWt(child.mol), child.unresolved))
                material.mmol += need
            else:
                inputs.append(StepInput(child.smiles, need, visit(child, need / STEP_YIELD)))
        steps.append(PlanStep(len(steps) + 1, node.smiles, recipe, inputs))
        for warning in recipe.warnings:
            warnings.append(f"Step {len(steps)}: {warning}")
        return len(steps)

    visit(root, scale_mmol)
    if any(m.unresolved for m in materials.values()):
        warnings.append("At least one precursor is complex and could not be broken down further within the "
                        "step limit; increase the maximum steps or prepare it by another route.")
    final = steps[-1].recipe
    return MultiStepRecipe(
        target_smiles=final.target_smiles, formula=final.formula, mw=final.mw, exact_mass=final.exact_mass,
        scale_mmol=scale_mmol, steps=steps, starting_materials=list(materials.values()),
        longest_sequence=root.depth, warnings=warnings)
