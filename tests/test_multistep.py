import pytest
from rdkit import Chem

from chemgen import multistep, synthesis

# amide <- (reductive amination) acid: a genuine two-step target
TWO_STEP = "O=C(NCc1ccc(F)cc1)c1ccc(CN2CCN(c3ncccn3)CC2)cc1"
ACID = "O=C(O)c1ccc(CN2CCN(c3ncccn3)CC2)cc1"
ALDEHYDE = "O=Cc1ccc(C(=O)O)cc1"
PIPERAZINE = "c1cnc(N2CCNCC2)nc1"
BENZYLAMINE = "NCc1ccc(F)cc1"


def plan(smiles, steps=3, scale=1.0):
    return multistep.propose_plan(Chem.MolFromSmiles(smiles), scale, steps)


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_simple_target_is_a_single_step_plan():
    p = plan("CC(=O)Nc1ccc(O)cc1")
    assert p.n_steps == 1 and p.longest_sequence == 1 and p.steps[0].recipe.template_key == "amide"
    assert all(i.from_step is None for i in p.steps[0].inputs)


def test_complex_target_is_planned_in_two_ordered_steps():
    p = plan(TWO_STEP)
    assert p.n_steps == 2 and p.longest_sequence == 2
    first, second = p.steps
    assert (first.recipe.template_key, second.recipe.template_key) == ("reductive_amination", "amide")
    assert canon(first.product) == canon(ACID)                       # step 1 makes the acid used in step 2
    assert canon(second.product) == canon(TWO_STEP)
    from_first = [i for i in second.inputs if i.from_step == 1]
    assert len(from_first) == 1 and canon(from_first[0].smiles) == canon(ACID)


def test_earlier_steps_are_scaled_so_later_steps_get_what_they_need():
    p = plan(TWO_STEP, scale=2.0)
    first, second = p.steps
    acid_needed = next(i for i in second.inputs if i.from_step == 1).mmol
    assert acid_needed == pytest.approx(2.0)                          # acid is limiting in the amide step
    assert first.recipe.scale_mmol == pytest.approx(acid_needed / multistep.STEP_YIELD)
    materials = {canon(m.smiles): m.mmol for m in p.starting_materials}
    assert materials[canon(ALDEHYDE)] == pytest.approx(2.5)           # limiting reagent of step 1
    assert materials[canon(PIPERAZINE)] == pytest.approx(2.5 * 1.1)
    assert materials[canon(BENZYLAMINE)] == pytest.approx(2.0 * 1.1)
    assert canon(ACID) not in materials                               # intermediates are not bought


def test_step_limit_leaves_a_flagged_complex_precursor():
    p = plan(TWO_STEP, steps=1)
    assert p.n_steps == 1 and any(m.unresolved for m in p.starting_materials)
    assert any("could not be broken down" in w for w in p.warnings)
    assert "complex precursor" in p.to_markdown()


def test_no_plan_when_no_reaction_class_applies_or_building_blocks_are_screened():
    assert plan("CC(=O)Oc1ccccc1C(=O)O") is None                      # aryl ester: no template
    assert plan("C[C@@H](Cc1cccc(F)c1)NC(=O)c1ccccn1") is None        # only route needs an amphetamine-type amine


def test_step_limit_is_clamped_to_a_sane_range():
    assert plan(TWO_STEP, steps=99).n_steps == 2
    assert plan("CC(=O)Nc1ccc(O)cc1", steps=0).n_steps == 1


def test_markdown_has_overview_materials_inputs_and_every_step():
    text = plan(TWO_STEP).to_markdown()
    for part in ["Route overview", "Starting materials", "Inputs for step 1", "Inputs for step 2",
                 "## Step 1 of 2: Reductive amination", "## Step 2 of 2: Amide coupling",
                 "product of step 1", "Planning assumption", "**Procedure**", "**Work-up**", "unknown toxicity"]:
        assert part in text, part
    assert text.index("Step 1 of 2") < text.index("Step 2 of 2")


def test_summary_names_the_steps_in_order():
    assert plan(TWO_STEP).summary == "2 steps: Reductive amination (NaBH(OAc)3) → Amide coupling (HATU)"
    assert plan("CC(=O)Nc1ccc(O)cc1").summary.startswith("1 step: ")


def test_starting_material_check_marks_known_and_unknown_compounds():
    p = plan("CC(=O)Nc1ccc(O)cc1")
    p.annotate(lambda smi: 176 if canon(smi) == canon("CC(=O)O") else None)
    status = {canon(m.smiles): (m.pubchem_cid, m.checked) for m in p.starting_materials}
    assert status[canon("CC(=O)O")] == (176, True) and status[canon("Nc1ccc(O)cc1")] == (None, True)
    text = p.to_markdown()
    assert "PubChem CID 176" in text and "not found in PubChem" in text


def test_starting_material_check_survives_a_failing_lookup():
    p = plan("CC(=O)Nc1ccc(O)cc1")

    def offline(smiles):
        raise ConnectionError("offline")

    p.annotate(offline)
    assert not any(m.checked for m in p.starting_materials) and "availability not checked" in p.to_markdown()


def test_acid_in_the_amine_partner_is_a_chemoselectivity_conflict_for_sulfonamides():
    routes = synthesis.propose_routes(Chem.MolFromSmiles("CS(=O)(=O)Nc1ccc(C(=O)O)cc1"))
    assert routes[0].template.key == "sulfonamide" and routes[0].conflicts == 1


def test_every_step_recipe_is_a_full_procedure():
    for step in plan(TWO_STEP).steps:
        assert step.recipe.steps and step.recipe.workup and step.recipe.hazards and step.recipe.reagents
