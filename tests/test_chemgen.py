import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

from chemgen import chem, safety
from chemgen.generator import ConditionalSmilesLM, Generator, train_generator
from chemgen.pipeline import DesignPipeline, DesignTargets, desirability
from chemgen.predictor import PropertyPredictor, _scaffold_split
from chemgen.tokenizer import Vocab, tokenize

DRUGLIKE = [
    "CC(=O)Oc1ccccc1C(=O)O",                   # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",              # ibuprofen
    "CN1CCC[C@H]1c1cccnc1",                    # nicotine
    "CC(=O)Nc1ccc(O)cc1",                      # paracetamol
    "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",           # melatonin
    "O=C(O)c1ccccc1O",                         # salicylic acid
    "Cc1ccc(cc1)S(=O)(=O)N",                   # tosylamide
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",              # procainamide
]


def test_tokenizer_round_trip():
    for smiles in ["CC(=O)Oc1ccccc1C(=O)Cl", "C[C@@H](Br)[C@H](F)CO", "C1CC%10CC1%10", "[nH]1cccc1"]:
        vocab = Vocab.build([smiles])
        assert vocab.decode(vocab.encode(smiles) + [vocab.eos_id, vocab.pad_id]) == smiles


def test_tokenizer_rejects_unknown_characters():
    with pytest.raises(ValueError):
        tokenize("CC!C")


@pytest.mark.parametrize("smiles", DRUGLIKE[:5])
def test_safety_accepts_ordinary_drugs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    verdict = safety.assess(mol)
    assert verdict.passed, verdict.reasons


@pytest.mark.parametrize("smiles, reason", [
    ("CCOP(=O)(OCC)c1ccccc1", "disallowed elements"),     # phosphorus
    ("[Na+].CC(=O)[O-]", "multiple fragments"),
    ("S=C1SC(=Cc2ccccc2)C(=O)N1", "structural alert"),    # benzylidene rhodanine, a PAINS motif
    ("C" * 60, "molecular weight"),
])
def test_safety_rejects(smiles, reason):
    verdict = safety.assess(Chem.MolFromSmiles(smiles))
    assert not verdict.passed
    assert any(reason in r for r in verdict.reasons), verdict.reasons


def test_basic_properties_match_rdkit():
    props = chem.basic_properties(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert props["MW"] == pytest.approx(180.16, abs=0.05)
    assert 0 < props["QED"] < 1 and 1 <= props["SAS"] <= 10


def test_scaffold_split_keeps_scaffolds_disjoint():
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mols = [Chem.MolFromSmiles(s) for s in DRUGLIKE * 3]
    train, test = _scaffold_split(mols, test_frac=0.25)
    assert len(set(train) & set(test)) == 0 and len(train) + len(test) == len(mols)
    scaffold = lambda i: MurckoScaffold.MurckoScaffoldSmiles(mol=mols[i])
    assert {scaffold(i) for i in train}.isdisjoint({scaffold(i) for i in test})


def test_predictor_learns_and_reports_uncertainty(tmp_path):
    rng = np.random.default_rng(0)
    pool = ["C" * n for n in range(2, 14)] + ["c1ccccc1" + "C" * n for n in range(0, 10)] + DRUGLIKE
    smiles = list(rng.choice(pool, size=120))
    y = [chem.basic_properties(Chem.MolFromSmiles(s))["logP"] for s in smiles]
    pd.DataFrame({"smiles": smiles, "measured log solubility in mols per litre": y}).to_csv(
        tmp_path / "delaney.csv", index=False)

    predictor, metrics = PropertyPredictor.train("solubility", tmp_path, n_estimators=50)
    assert metrics["rmse"] < metrics["baseline_rmse"]

    mols = [Chem.MolFromSmiles(s) for s in DRUGLIKE]
    mean, std = predictor.predict(mols)
    assert mean.shape == std.shape == (len(mols),) and (std >= 0).all()

    predictor.save(tmp_path / "p.joblib")
    reloaded = PropertyPredictor.load(tmp_path / "p.joblib")
    assert np.allclose(reloaded.predict(mols)[0], mean)


def unique_molecules(n_chain=25, n_tail=26):
    """650 distinct, simple molecules: alkyl aryl ethers with different chain lengths."""
    return ["C" * a + "Oc1ccccc1" + "C" * b for a in range(1, n_chain + 1) for b in range(n_tail)]


def test_generator_trains_saves_and_samples(tmp_path):
    smiles = unique_molecules()  # train_generator holds out at least 500 molecules for validation
    props = [chem.basic_properties(Chem.MolFromSmiles(s)) for s in smiles]
    df = pd.DataFrame({"smiles": smiles, "logP": [p["logP"] for p in props],
                       "qed": [p["QED"] for p in props], "SAS": [p["SAS"] for p in props]})
    df.to_csv(tmp_path / "zinc.csv", index=False)

    train_generator(tmp_path / "zinc.csv", tmp_path / "g.pt", tmp_path / "train.txt",
                    n_mols=len(df), epochs=1, batch_size=64, log=lambda *_: None)
    gen = Generator.load(tmp_path / "g.pt")
    samples = gen.sample(logp=1.0, qed=0.7, sas=2.0, n=20)
    assert len(samples) == 20 and all(isinstance(s, str) for s in samples)
    assert (tmp_path / "train.txt").read_text().splitlines()


def test_extra_training_files_are_mixed_in_and_duplicates_dropped(tmp_path):
    smiles = unique_molecules()
    props = [chem.basic_properties(Chem.MolFromSmiles(s)) for s in smiles]
    pd.DataFrame({"smiles": smiles, "logP": [p["logP"] for p in props],
                  "qed": [p["QED"] for p in props], "SAS": [p["SAS"] for p in props]}).to_csv(
        tmp_path / "zinc.csv", index=False)
    new, dup = "CCNC(=O)c1ccc(F)cc1", smiles[0]
    pd.DataFrame({"smiles": [new, dup], "logP": [1.9, 2.0], "qed": [0.8, 0.6], "SAS": [1.5, 1.2],
                  "source": ["pubchem", "pubchem"], "source_id": ["1", "2"],
                  "fetched_at": ["2026-01-01"] * 2}).to_csv(tmp_path / "harvested.csv", index=False)

    train_generator(tmp_path / "zinc.csv", tmp_path / "g.pt", tmp_path / "train.txt", n_mols=10_000,
                    epochs=1, batch_size=64, log=lambda *_: None, extra_csvs=[tmp_path / "harvested.csv"])
    trained = (tmp_path / "train.txt").read_text().splitlines()
    assert Chem.MolToSmiles(Chem.MolFromSmiles(new)) in trained
    assert len(trained) == len(set(trained)) == len(smiles) + 1


def test_sampling_is_reproducible_with_seed():
    vocab = Vocab.build(DRUGLIKE)
    gen = Generator(ConditionalSmilesLM(len(vocab), 3, hidden=32, emb=16), vocab,
                    np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    torch.manual_seed(7)
    first = gen.sample(0, 0, 0, n=10)
    torch.manual_seed(7)
    assert gen.sample(0, 0, 0, n=10) == first


def test_desirability_prefers_on_target_molecules():
    t = DesignTargets(logp=2.5, qed=0.8, sas=3.0, min_logs=-4.0)
    df = pd.DataFrame({
        "logP": [2.5, 5.5, 2.5], "QED": [0.9, 0.9, 0.4], "SAS": [2.0, 2.0, 2.0],
        "pred_logS": [-3.0, -3.0, -3.0], "brenk_alerts": [0, 0, 0],
    })
    good, off_logp, low_qed = desirability(df, t)
    assert good > off_logp and good > low_qed
    df["brenk_alerts"] = 2
    assert desirability(df, t)[0] < good


def test_diverse_top_skips_near_duplicates():
    smiles = ["c1ccccc1CCO", "c1ccccc1CCCO", "CC(=O)Nc1ccc(O)cc1", "CCN(CC)CCNC(=O)c1ccc(N)cc1"]
    df = pd.DataFrame({"SMILES": smiles, "score": [0.9, 0.8, 0.7, 0.6]})
    mols = {s: Chem.MolFromSmiles(s) for s in smiles}
    picked = DesignPipeline._diverse_top(df, mols, top_k=3, max_similarity=0.5)
    assert smiles[0] in set(picked["SMILES"]) and len(picked) == 3
