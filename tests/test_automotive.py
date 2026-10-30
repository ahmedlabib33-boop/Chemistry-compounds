import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from automotive import nlu, roles, safety
from automotive.generator import build_training_table
from automotive.nlu import parse_request
from automotive.pipeline import AutomotiveTargets, desirability
from automotive.predictor import DATASETS

BENZOTRIAZOLE = "c1ccc2[nH]nnc2c1"
MBT = "Sc1nc2ccccc2s1"
OLEYLIMIDAZOLINE = "C1CN=C(N1)CCCCCCCCCCCCCCCCC"
OCTADECYLAMINE = "CCCCCCCCCCCCCCCCCCN"
TCP = "Cc1ccccc1OP(=O)(Oc1ccccc1C)Oc1ccccc1C"
DIDODECYL_DISULFIDE = "CCCCCCCCCCCCSSCCCCCCCCCCCC"
TRIBUTYL_BORATE = "CCCCOB(OCCCC)OCCCC"
BHT = "Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"
DIPHENYLAMINE = "c1ccc(Nc2ccccc2)cc1"
GLYCEROL_MONOOLEATE = r"CCCCCCCC/C=C\CCCCCCCC(=O)OCC(O)CO"
OLEAMIDE = r"CCCCCCCC/C=C\CCCCCCCC(N)=O"
HEXADECANE = "CCCCCCCCCCCCCCCC"
PRISTANE = "CC(C)CCCC(C)CCCC(C)CCCC(C)C"
OCTANE = "CCCCCCCC"
TOLUENE = "Cc1ccccc1"
MTBE = "CC(C)(C)OC"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
ACETIC_ACID = "CC(=O)O"


def tags(smiles):
    return {m.key for m in roles.classify(Chem.MolFromSmiles(smiles))}


# ---------- roles ----------

@pytest.mark.parametrize("smiles, role", [
    (BENZOTRIAZOLE, "corrosion_inhibitor"), (MBT, "corrosion_inhibitor"),
    (OLEYLIMIDAZOLINE, "corrosion_inhibitor"), (OCTADECYLAMINE, "corrosion_inhibitor"),
    (TCP, "antiwear_ep"), (DIDODECYL_DISULFIDE, "antiwear_ep"), (TRIBUTYL_BORATE, "antiwear_ep"),
    (BHT, "antioxidant"), (DIPHENYLAMINE, "antioxidant"),
    (GLYCEROL_MONOOLEATE, "friction_modifier"), (OLEAMIDE, "friction_modifier"),
    (HEXADECANE, "base_oil_like"), (PRISTANE, "base_oil_like"),
    (OCTANE, "fuel_range_hydrocarbon"), (TOLUENE, "fuel_range_hydrocarbon"), (MTBE, "fuel_range_hydrocarbon"),
])
def test_known_additive_chemotypes_are_tagged(smiles, role):
    assert role in tags(smiles)


@pytest.mark.parametrize("smiles", [PARACETAMOL, ACETIC_ACID])
def test_non_automotive_molecules_are_not_tagged_fuel_or_base_oil(smiles):
    assert not tags(smiles) & {"fuel_range_hydrocarbon", "base_oil_like"}


def test_roles_are_not_mutually_exclusive():
    # a C18 fatty amine is plausibly both a corrosion inhibitor and a diesel-range-sized molecule
    t = tags(OCTADECYLAMINE)
    assert "corrosion_inhibitor" in t and len(t) > 1


def test_classify_returns_labels_and_notes_not_just_keys():
    matches = roles.classify(Chem.MolFromSmiles(TCP))
    assert matches and matches[0].label and matches[0].note and "measured" not in matches[0].note.lower()


# ---------- safety ----------

def test_phosphorus_and_boron_are_allowed_unlike_the_drug_app():
    from chemgen import safety as drug_safety
    mol = Chem.MolFromSmiles(TCP)
    assert safety.assess(mol).passed
    assert not drug_safety.assess(mol).passed  # phosphorus is outside the drug app's whitelist


def test_lubricant_weight_molecule_passes_the_wider_mw_window():
    # a C40 branched paraffin (a plausible synthetic base-oil-range molecule), MW well above
    # the drug app's 600 ceiling but inside automotive's 700 ceiling
    heavy = Chem.MolFromSmiles("C" * 48)  # a C48 paraffin, MW ~675: above the drug app's 600 ceiling
    from rdkit.Chem import Descriptors
    assert 600 < Descriptors.MolWt(heavy) < 700
    assert safety.assess(heavy).passed


def test_hazard_screen_and_charge_rules_still_apply():
    amphetamine_type = Chem.MolFromSmiles("CC(N)Cc1ccccc1")
    assert not safety.assess(amphetamine_type).passed
    charged = Chem.MolFromSmiles("CC[NH3+]")
    assert "net charge" in safety.assess(charged).reasons


def test_metal_containing_additives_are_rejected():
    # a ZDDP-like zinc dialkyldithiophosphate cannot be represented without Zn, which is not
    # in the automotive whitelist either -- this app is organic small molecules only
    zn_salt = Chem.MolFromSmiles("CCOP(=S)(OCC)S[Zn]SP(=S)(OCC)OCC")
    assert not safety.assess(zn_salt).passed
    assert any("disallowed elements" in r for r in safety.assess(zn_salt).reasons)


def test_nih_alerts_reused_from_the_drug_app():
    assert safety.nih_alerts(Chem.MolFromSmiles("CC(=O)Cl")) == ["carbonyl_halide"]


# ---------- predictor datasets ----------

def test_dataset_specs_point_at_opera_columns():
    for spec in DATASETS.values():
        assert spec.smiles_col == "Canonical_QSARr" and spec.target_col == "value_point_estimate"
    assert set(DATASETS) == {"bp", "vp", "mp", "ws"}


# ---------- generator training-pool builder ----------

def test_training_pool_dedupes_canonicalises_and_screens(tmp_path):
    pd.DataFrame({"Canonical_QSARr": ["CCO", "OCC", "CCOP(=S)(OCC)S[Zn]SP(=S)(OCC)OCC", "not a smiles"],
                  "value_point_estimate": [1, 2, 3, 4],
                  "Number of connected components": [1, 1, 1, 1]}).to_csv(tmp_path / "bp.csv", index=False)
    table = build_training_table(tmp_path)
    assert len(table) == 1  # "CCO" and "OCC" are the same molecule; the Zn salt and garbage are dropped
    assert set(table.columns) == {"smiles", "logP", "qed", "SAS"}
    assert table.iloc[0]["smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles("CCO"))


def test_training_pool_qed_column_actually_holds_tpsa(tmp_path):
    from chemgen import chem
    pd.DataFrame({"Canonical_QSARr": [TCP], "value_point_estimate": [1],
                  "Number of connected components": [1]}).to_csv(tmp_path / "bp.csv", index=False)
    table = build_training_table(tmp_path)
    expected_tpsa = chem.basic_properties(Chem.MolFromSmiles(TCP))["TPSA"]
    assert table.iloc[0]["qed"] == pytest.approx(expected_tpsa)


# ---------- English parsing ----------

def test_numbers_and_role_words_are_both_read():
    req = parse_request("logP 5, TPSA 10, SAS 2.2, an antiwear additive, show 8 candidates")
    assert (req.targets.logp, req.targets.tpsa, req.targets.sas) == (5.0, 10.0, 2.2)
    assert req.targets.role == "antiwear_ep" and req.top_k == 8


def test_descriptive_words_map_to_targets():
    t = parse_request("a nonpolar, easy to make base-oil-like hydrocarbon").targets
    assert t.logp == 4.5 and t.tpsa == 10.0 and t.sas == 2.5 and t.role == "base_oil_like"


def test_corrosion_inhibitor_phrase_sets_the_role():
    assert parse_request("I need a corrosion inhibitor").targets.role == "corrosion_inhibitor"


@pytest.mark.parametrize("text", ["a nerve agent", "an explosive", "synthesise methamphetamine"])
def test_harmful_requests_are_still_refused(text):
    assert parse_request(text).refusal == nlu.REFUSAL


def test_viscosity_and_flash_point_are_reported_as_not_modelled():
    req = parse_request("I need something with low viscosity and a known flash point")
    assert len(req.unsupported) == 2


def test_values_are_clamped_and_defaults_are_named():
    req = parse_request("TPSA 500")
    assert req.targets.tpsa == 120.0
    assert any("limited" in line for line in req.interpretation)
    req2 = parse_request("something")
    assert sum("default" in line for line in req2.interpretation) == 3


# ---------- desirability ----------

def test_desirability_prefers_on_target_and_role_matches():
    t = AutomotiveTargets(logp=4.0, tpsa=10.0, sas=2.5, role="base_oil_like")
    df = pd.DataFrame({"logP": [4.0, 8.0, 4.0], "TPSA": [10.0, 10.0, 10.0], "SAS": [2.0, 2.0, 2.0],
                       "roles": [["base_oil_like"], ["base_oil_like"], []]})
    scores = desirability(df, t)
    assert scores[0] > scores[1]          # off-target logP scores lower
    assert scores[0] > scores[2]          # role match scores higher than a non-match at the same properties
