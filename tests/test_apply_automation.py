import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from chemgen import automation, chem, prepare, safety
from chemgen.apply import ModelBundle, analyze
from chemgen.automation import CycleConfig, Registry, decide, evaluate_generator, run_cycle, run_forever
from chemgen.predictor import PropertyPredictor

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PHOSPHONATE = "CCOP(=O)(OCC)c1ccccc1"
MUSTARD = "ClCCN(CCCl)c1ccccc1"


# ---------- safety / chem additions ----------

def test_mustards_are_a_hazard_pattern_and_nih_alerts_are_separate_from_pains():
    assert "nitrogen / sulfur mustard" in safety.hazard_flags(Chem.MolFromSmiles(MUSTARD))
    assert safety.nih_alerts(Chem.MolFromSmiles("CC(=O)Cl")) == ["carbonyl_halide"]
    assert safety.nih_alerts(Chem.MolFromSmiles(PARACETAMOL)) == []


def test_standardize_drops_salts_and_neutralises():
    mol = chem.standardize(Chem.MolFromSmiles("[Na+].[Cl-].COc1ccc([C@H]([NH3+])c2ccc(F)cc2)cc1"))
    assert Chem.GetFormalCharge(mol) == 0 and len(Chem.GetMolFrags(mol)) == 1


# ---------- apply ----------

@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("apply")
    pool = ["C" * n for n in range(2, 14)] + [f"c1ccccc1{'C' * n}" for n in range(10)] + [PARACETAMOL, ASPIRIN]
    smiles = [pool[i % len(pool)] for i in range(120)]
    y = [chem.basic_properties(Chem.MolFromSmiles(s))["logP"] for s in smiles]
    pd.DataFrame({"smiles": smiles, "measured log solubility in mols per litre": y}).to_csv(tmp / "delaney.csv", index=False)
    predictor, _ = PropertyPredictor.train("solubility", tmp, n_estimators=30)
    fps = [chem.fingerprint(Chem.MolFromSmiles(s)) for s in set(smiles)]
    return ModelBundle(predictor, predictor, fps)


def test_ordinary_molecule_gets_properties_predictions_domain_and_recipe(bundle):
    result = analyze(PARACETAMOL, bundle)
    assert result.status == "ok" and result.smiles == PARACETAMOL
    assert set(result.predictions) == {"logS", "logD"} and all(sd >= 0 for _, sd in result.predictions.values())
    assert result.nearest_similarity == pytest.approx(1.0) and result.in_domain
    assert any("training data" in n for n in result.notes)       # in-sample prediction is flagged
    assert result.recipe is not None and "Amide coupling" in result.to_markdown()


def test_far_from_training_data_is_flagged_out_of_domain(bundle):
    result = analyze("O=C(Nc1ncccn1)c1ccc2c(c1)OCO2", bundle)
    assert result.status == "ok" and result.in_domain is False
    assert "unreliable" in result.to_markdown()


def test_salts_are_stripped_before_analysis(bundle):
    assert analyze(f"{PARACETAMOL}.[Na+].[Cl-]", bundle).smiles == PARACETAMOL


def test_small_molecule_outside_design_space_is_still_analysed_with_a_note(bundle):
    result = analyze("CCCCCC", bundle)
    assert result.status == "ok" and any("molecular weight" in n for n in result.notes)


def test_hazard_matches_are_refused_with_no_output(bundle):
    result = analyze(MUSTARD, bundle)
    assert result.status == "refused" and not result.predictions and result.recipe is None
    text = result.to_markdown()
    assert "Refused" in text and "ML predictions" not in text and "Reagents" not in text


def test_unsupported_elements_and_garbage_are_handled(bundle):
    assert analyze(PHOSPHONATE, bundle).status == "unsupported"
    assert analyze("definitely not a molecule", bundle).status == "invalid"
    assert analyze("", bundle).status == "invalid"


def test_reactive_alert_keeps_predictions_but_withholds_the_recipe(bundle):
    result = analyze("C=CC(=O)c1ccccc1", bundle)       # enone: NIH acrylate alert
    assert result.status == "ok" and result.predictions and result.recipe is None
    assert "withheld" in result.recipe_note and "ML predictions" in result.to_markdown()


def test_molecule_without_a_supported_route_says_so(bundle):
    assert "multi-step" in analyze("CCCCCC", bundle).to_markdown()


# ---------- registry ----------

def make_model_dir(path: Path, content: str, meta: dict | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "generator.pt").write_text(content)
    (path / "train_smiles.txt").write_text(content + "-smiles")
    if meta is not None:
        (path / automation.META_FILE).write_text(json.dumps(meta))
    return path


def test_labels_never_collide_even_within_one_second(tmp_path):
    live = make_model_dir(tmp_path / "models", "v0")
    reg = Registry(live)
    labels = {reg.new_label() for _ in range(3)}
    assert len(labels) == 1                              # nothing registered yet, so same label is fine
    first = reg.promote(make_model_dir(tmp_path / "c1", "v1"), {"version": reg.new_label()})
    second = reg.promote(make_model_dir(tmp_path / "c2", "v2"), {"version": reg.new_label()})
    assert first != second and set(reg.versions()) == {"initial", first}
    assert (live / "registry" / first / "generator.pt").read_text() == "v1"      # not overwritten


def test_promote_archives_the_old_model_and_installs_the_new_one(tmp_path):
    live = make_model_dir(tmp_path / "models", "old")
    reg = Registry(live)
    assert reg.live_version() == "initial"
    version = reg.promote(make_model_dir(tmp_path / "cand", "new"), {"version": "v2", "metrics": {"x": 1}})
    assert version == "v2" and reg.live_version() == "v2"
    assert (live / "generator.pt").read_text() == "new"
    assert (live / "registry" / "initial" / "generator.pt").read_text() == "old"
    assert automation.read_meta(live)["previous_version"] == "initial"
    assert not list(live.glob("*.tmp"))


def test_rollback_walks_back_one_version_at_a_time(tmp_path):
    live = make_model_dir(tmp_path / "models", "v0")
    reg = Registry(live)
    reg.promote(make_model_dir(tmp_path / "c1", "v1"), {"version": "v1"})
    reg.promote(make_model_dir(tmp_path / "c2", "v2"), {"version": "v2"})
    assert reg.rollback() == "v1" and (live / "generator.pt").read_text() == "v1"
    assert reg.rollback() == "initial" and (live / "generator.pt").read_text() == "v0"
    with pytest.raises(RuntimeError):
        reg.rollback()
    assert set(reg.versions()) == {"initial", "v1", "v2"}      # nothing was lost


def test_rollback_with_nothing_to_restore_fails_cleanly(tmp_path):
    with pytest.raises(RuntimeError):
        Registry(tmp_path / "empty").rollback()


# ---------- evaluation and the promotion rule ----------

class FakeGenerator:
    def __init__(self, smiles):
        self.smiles = smiles

    def sample(self, logp, qed, sas, n=1000, **_):
        return list(self.smiles)


def test_evaluate_counts_each_funnel_stage():
    garbage, hexyl = "garbage", "CCCCCCOc1ccccc1"       # a long chain would trip an NIH alert
    gen = FakeGenerator([PARACETAMOL, PARACETAMOL, garbage, ASPIRIN, PHOSPHONATE, hexyl])
    props = chem.basic_properties(Chem.MolFromSmiles(PARACETAMOL))
    metrics = evaluate_generator(gen, known={Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))},
                                 conditions=[(props["logP"], props["QED"], props["SAS"])], n=6)
    assert {k: metrics[k] for k in ("sampled", "valid", "unique", "novel", "safe", "useful")} == \
        dict(sampled=6, valid=5, unique=4, novel=3, safe=2, useful=1)
    assert metrics["validity"] == pytest.approx(5 / 6) and metrics["useful_yield"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("cand, live, expect", [
    ({"useful_yield": 0.30, "validity": 0.80}, None, True),
    ({"useful_yield": 0.30, "validity": 0.80}, {"useful_yield": 0.20, "validity": 0.80}, True),
    ({"useful_yield": 0.202, "validity": 0.80}, {"useful_yield": 0.20, "validity": 0.80}, False),   # tiny gain
    ({"useful_yield": 0.15, "validity": 0.80}, {"useful_yield": 0.20, "validity": 0.80}, False),    # worse
    ({"useful_yield": 0.40, "validity": 0.70}, {"useful_yield": 0.20, "validity": 0.80}, False),    # validity drop
])
def test_promotion_rule(cand, live, expect):
    assert decide(cand, live)[0] is expect


# ---------- the cycle ----------

def cycle_env(tmp_path, live: str | None = "live", harvested_rows: int = 10, **cfg_kw):
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir(); models.mkdir()
    (data / "harvested.csv").write_text("smiles\n" + "\n".join(["x"] * harvested_rows) + ("\n" if harvested_rows else ""))
    if live:
        make_model_dir(models, live)
    calls = {"harvest": 0, "train": 0}

    def harvest_fn(cfg):
        calls["harvest"] += 1
        with (data / "harvested.csv").open("a") as f:
            f.write("y\n" * 5)

    def train_fn(cfg, out_dir):
        calls["train"] += 1
        make_model_dir(out_dir, "candidate")
        return {"best_val_loss": 0.5}

    def evaluate_fn(model_dir, known, cfg):
        better = (Path(model_dir) / "generator.pt").read_text() == "candidate"
        return {"useful_yield": 0.4 if better else 0.1, "validity": 0.8}

    cfg = CycleConfig(data_dir=data, model_dir=models, min_new=5, **cfg_kw)
    return cfg, models, calls, dict(harvest_fn=harvest_fn, train_fn=train_fn, evaluate_fn=evaluate_fn, log=lambda *_: None)


def test_cycle_promotes_a_better_candidate_and_logs_it(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path)
    report = run_cycle(cfg, **fns)
    assert report.promoted and report.retrained and calls == {"harvest": 1, "train": 1}
    assert (models / "generator.pt").read_text() == "candidate"
    assert automation.read_meta(models)["harvested_rows"] == 15           # 10 + 5 harvested this cycle
    log = automation.history(models)
    assert len(log) == 1 and log[0]["promoted"] and log[0]["candidate_metrics"]["useful_yield"] == 0.4
    assert [s["step"] for s in log[0]["steps"]] == ["harvest", "gate", "train", "evaluate", "promote"]


def test_cycle_skips_retraining_without_enough_new_data(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path)
    run_cycle(cfg, **fns)
    report = run_cycle(CycleConfig(**{**cfg.__dict__, "skip_harvest": True}), **fns)
    assert not report.retrained and calls["train"] == 1 and "not enough new data" in report.decision


def test_force_retrains_but_an_equal_candidate_is_not_promoted(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path, live="candidate")     # live already as good as any candidate
    report = run_cycle(CycleConfig(**{**cfg.__dict__, "force": True, "skip_harvest": True}), **fns)
    assert report.retrained and not report.promoted and "below the required" in report.decision


def test_first_run_with_no_live_model_always_trains_and_installs(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path, live=None, harvested_rows=0)
    report = run_cycle(cfg, **fns)
    assert report.promoted and report.live_metrics is None and (models / "generator.pt").exists()


def test_no_promote_flag_keeps_the_live_model(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path, promote=False)
    report = run_cycle(cfg, **fns)
    assert report.retrained and not report.promoted and (models / "generator.pt").read_text() == "live"


def test_a_failed_harvest_does_not_stop_the_cycle(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path)
    fns["harvest_fn"] = lambda cfg: (_ for _ in ()).throw(ConnectionError("offline"))
    report = run_cycle(cfg, **fns)
    harvest = report.steps[0]
    assert harvest["status"] == "failed" and "offline" in harvest["detail"]
    assert report.retrained                                             # 10 existing rows >= min_new


def test_a_failed_training_leaves_the_live_model_untouched(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path)
    fns["train_fn"] = lambda cfg, out: (_ for _ in ()).throw(RuntimeError("out of memory"))
    report = run_cycle(cfg, **fns)
    assert not report.promoted and "training failed" in report.decision
    assert (models / "generator.pt").read_text() == "live" and len(automation.history(models)) == 1


def test_run_forever_sleeps_between_cycles_and_respects_max_cycles(tmp_path):
    cfg, models, calls, fns = cycle_env(tmp_path)
    sleeps = []
    reports = run_forever(cfg, interval_hours=2, max_cycles=3, sleep=sleeps.append, **fns)
    assert len(reports) == 3 and sleeps == [7200, 7200]


# ---------- prepare / status ----------

def test_status_lists_every_missing_piece_with_its_fix(tmp_path):
    checks = {c.name: c for c in prepare.run_checks(tmp_path / "data", tmp_path / "models")}
    assert not checks["Training data"].ok and checks["Training data"].fix == "python -m chemgen download"
    assert checks["Property predictors"].fix == "python -m chemgen train-predictors"
    assert checks["Molecule generator"].fix == "python -m chemgen train-generator"
    assert not prepare.is_ready(list(checks.values()))
    assert "python -m chemgen prepare" in prepare.render(list(checks.values()))


def test_prepare_runs_missing_steps_in_order_and_is_idempotent(tmp_path):
    data, models = tmp_path / "data", tmp_path / "models"
    order = []

    def make(files_dir, names):
        def run():
            files_dir.mkdir(parents=True, exist_ok=True)
            for n in names:
                (files_dir / n).write_text("x")
        return run

    def tracked(name, action):
        def run():
            order.append(name); action()
        return run

    runners = {"download": tracked("download", make(data, prepare.DATA_FILES)),
               "train-predictors": tracked("train-predictors", make(models, prepare.PREDICTOR_FILES)),
               "train-generator": tracked("train-generator", make(models, prepare.GENERATOR_FILES)),
               "harvest": tracked("harvest", make(data, ["harvested.csv"]))}
    done = prepare.prepare(data_dir=data, model_dir=models, harvest_budget=100, runners=runners, log=lambda *_: None)
    assert done == ["download", "train-predictors", "harvest", "train-generator"] == order   # harvest precedes generator
    assert prepare.is_ready(prepare.run_checks(data, models))
    assert prepare.prepare(data_dir=data, model_dir=models, runners=runners, log=lambda *_: None) == []
