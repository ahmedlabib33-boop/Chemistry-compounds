import pandas as pd
import pytest
from rdkit import Chem

from chemgen import chem, safety, synthesis
from chemgen.nlu import REFUSAL, parse_request
from chemgen.pipeline import DesignResult


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def best_route(smiles):
    routes = synthesis.propose_routes(Chem.MolFromSmiles(smiles))
    return routes[0] if routes else None


# ---------- English parsing ----------

def test_numbers_override_words_and_are_read():
    req = parse_request("drug-like molecule, logP of 3.2, QED 0.9, SAS 2.2, logS -3.5, give me 12 molecules")
    t = req.targets
    assert (t.logp, t.qed, t.sas, t.min_logs) == (3.2, 0.9, 2.2, -3.5)
    assert req.top_k == 12 and req.refusal is None


def test_negative_logp_is_not_stripped_of_its_sign():
    assert parse_request("logP around -0.5").targets.logp == -0.5


def test_descriptive_words_map_to_targets():
    t = parse_request("an easy to make, water soluble, lipophilic and highly drug-like compound").targets
    assert t.sas == 2.5 and t.min_logs == -3.0 and t.logp == 3.5 and t.qed == 0.9


def test_moderately_lipophilic_beats_lipophilic():
    assert parse_request("moderately lipophilic").targets.logp == 2.5


def test_values_are_clamped_to_slider_range():
    req = parse_request("logP 12")
    assert req.targets.logp == 6.0
    assert any("limited" in line for line in req.interpretation)


def test_unspecified_targets_use_defaults_and_say_so():
    req = parse_request("something nice")
    assert req.targets.logp == 2.5
    assert sum("default" in line for line in req.interpretation) == 4


def test_unsupported_requests_are_reported_not_dropped():
    req = parse_request("a kinase inhibitor with molecular weight 300 containing fluorine")
    assert len(req.unsupported) == 3


@pytest.mark.parametrize("text", [
    "a nerve agent", "design a poison", "an explosive compound", "like fentanyl",
    "a toxic molecule", "synthesise methamphetamine", "a potent opioid"])
def test_harmful_requests_are_refused(text):
    req = parse_request(text)
    assert req.refusal == REFUSAL


@pytest.mark.parametrize("text", ["a non-toxic, low toxicity molecule", "methyl ester analogue of aspirin",
                                  "not toxic and water soluble"])
def test_benign_wording_is_not_refused(text):
    assert parse_request(text).refusal is None


# ---------- safety additions ----------

@pytest.mark.parametrize("smiles", [
    "CC(N)Cc1ccccc1",                       # amphetamine-type
    "CCC1(CC)C(=O)NC(=O)NC1=O",             # barbiturate
    "CC(=O)N(c1ccccc1)C1CCN(C)CC1",         # 4-anilidopiperidine scaffold
    "CC(C)N=[N+]=[N-]",                     # azide
    "c1cc([N+](=O)[O-])cc([N+](=O)[O-])c1"  # polynitro
])
def test_hazard_screen_flags(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert safety.hazard_flags(mol)
    assert not safety.assess(mol).passed


@pytest.mark.parametrize("smiles", ["CC(=O)Oc1ccccc1C(=O)O", "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
                                    "CC(=O)Nc1ccc(O)cc1"])
def test_hazard_screen_passes_ordinary_drugs(smiles):
    assert safety.hazard_flags(Chem.MolFromSmiles(smiles)) == []


def test_neutralize_removes_ammonium_and_charged_leftovers_are_rejected():
    ammonium = Chem.MolFromSmiles("COc1ccc([C@H]([NH3+])c2ccc(F)cc2)cc1")
    assert Chem.GetFormalCharge(chem.neutralize(ammonium)) == 0
    quaternary = Chem.MolFromSmiles("C[N+](C)(C)Cc1ccccc1CCCCCc1ccccc1")
    reasons = safety.assess(chem.neutralize(quaternary)).reasons
    assert "net charge" in reasons


# ---------- retrosynthesis ----------

@pytest.mark.parametrize("target, key, expected", [
    ("CC(=O)Nc1ccc(O)cc1", "amide", {"CC(=O)O", "Nc1ccc(O)cc1"}),
    ("CCNS(=O)(=O)c1ccc(C)cc1", "sulfonamide", {"Cc1ccc(S(=O)(=O)Cl)cc1", "CCN"}),
    ("CCN(CC)Cc1ccccc1", "reductive_amination", {"O=Cc1ccccc1", "CCNCC"}),
    ("c1ccnc(n1)N1CCCC1", "snar", {"Clc1ncccn1", "C1CCNC1"}),
    ("c1ccc(cc1)-c1ccncc1", "suzuki", {"Brc1ccncc1", "OB(O)c1ccccc1"}),
    ("CCCOc1ccccc1", "ether", {"CCCBr", "Oc1ccccc1"}),
    ("CCOC(=O)c1ccccc1", "ester", {"O=C(O)c1ccccc1", "CCO"}),
    ("CCNC(=O)Nc1ccccc1", "urea", {"CCN", "Nc1ccccc1"}),
    ("CC(C)(C)OC(=O)N1CCCC1", "boc", {"CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "C1CCNC1"}),
])
def test_each_reaction_class_disconnects_correctly(target, key, expected):
    route = best_route(target)
    assert route.template.key == key
    assert {canon(Chem.MolToSmiles(route.electrophile)), canon(Chem.MolToSmiles(route.nucleophile))} \
        == {canon(s) for s in expected}


def test_no_route_for_lactam_or_primary_amide_or_ammonia_disconnection():
    for smiles in ["O=C1CCCN1", "NC(=O)c1ccccc1", "Nc1ncccn1"]:
        assert synthesis.propose_routes(Chem.MolFromSmiles(smiles)) == []


def test_stereocentre_that_becomes_a_carbonyl_still_disconnects():
    route = best_route("C[C@H](Nc1ccc(CN)cc1)C1CC1")
    assert route.template.key == "reductive_amination"
    assert route.conflicts == 1  # the benzylic NH2 competes with the aniline NH2


def test_competing_groups_raise_cost():
    clean = best_route("CC(=O)Nc1ccccc1")
    clash = best_route("CC(=O)Nc1ccc(CN)cc1")  # free benzylic amine competes
    assert clean.conflicts == 0 and clash.conflicts == 1 and clash.cost > clean.cost


def test_routes_through_hazardous_building_blocks_are_dropped():
    # Amide of a 3-fluoroamphetamine-type amine: the only disconnection needs that amine.
    assert synthesis.propose_routes(Chem.MolFromSmiles("C[C@@H](Cc1cccc(F)c1)NC(=O)c1ccccn1")) == []


# ---------- recipes ----------

def test_recipe_quantities_follow_from_molecular_weights():
    recipe = synthesis.propose_recipe(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"), scale_mmol=2.0)
    rows = {name: (equiv, amount) for name, _, equiv, amount in recipe.reagents}
    assert rows["`CC(=O)O`"] == ("1", "2.00 mmol, 120 mg")          # acetic acid, MW 60.05
    assert rows["HATU"] == ("1.2", "2.40 mmol, 913 mg")             # MW 380.23
    assert rows["DIPEA"][1] == "6.00 mmol, 1045 µL"                 # 6 x 129.25 / 0.742
    assert recipe.formula == "C8H9NO2"
    assert recipe.exact_mass + synthesis.PROTON_MASS == pytest.approx(152.0706, abs=1e-3)


@pytest.mark.parametrize("target", [
    "CC(=O)Nc1ccc(O)cc1", "CCNS(=O)(=O)c1ccc(C)cc1", "CCN(CC)Cc1ccccc1", "c1ccnc(n1)N1CCCC1",
    "c1ccc(cc1)-c1ccncc1", "CCCOc1ccccc1", "CCOC(=O)c1ccccc1", "CCNC(=O)Nc1ccccc1", "CC(C)(C)OC(=O)N1CCCC1"])
def test_every_reaction_class_renders_a_complete_recipe(target):
    markdown = synthesis.propose_recipe(Chem.MolFromSmiles(target)).to_markdown()
    for section in ["**Reagents**", "**Procedure**", "**Work-up**", "**Purification:**",
                    "**Hazards and handling**", "unknown toxicity", "[M+H]+"]:
        assert section in markdown, section


def test_recipe_warns_about_stereochemistry_and_conflicts():
    recipe = synthesis.propose_recipe(Chem.MolFromSmiles("C[C@H](Nc1ccc(CN)cc1)C1CC1"))
    text = " ".join(recipe.warnings)
    assert "stereocentre" in text and "competing" in text


def test_no_recipe_when_no_template_applies():
    assert synthesis.propose_recipe(Chem.MolFromSmiles("CCCCO")) is None


def test_recipes_markdown_lists_every_molecule_and_handles_missing_routes():
    good, none = "CC(=O)Nc1ccc(O)cc1", "CCCCO"
    table = pd.DataFrame({"SMILES": [good, none]})
    recipes = {good: synthesis.propose_recipe(Chem.MolFromSmiles(good)), none: None}
    text = DesignResult(table, {}, recipes).recipes_markdown()
    assert "Molecule 1" in text and "Molecule 2" in text
    assert "No route from the supported" in text and "Amide coupling" in text
