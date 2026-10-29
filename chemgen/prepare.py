"""Readiness check and one-command preparation of the app.

``status`` reports what is installed, downloaded and trained and prints the exact command for
anything missing. ``prepare`` runs the missing steps in order. Both are idempotent.
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import DATA_DIR, MODEL_DIR

REQUIRED_PACKAGES = {"rdkit": "rdkit", "torch": "torch", "sklearn": "scikit-learn", "pandas": "pandas",
                     "numpy": "numpy", "joblib": "joblib", "streamlit": "streamlit"}
DATA_FILES = ("zinc250k.csv", "delaney.csv", "lipophilicity.csv")
PREDICTOR_FILES = ("solubility.joblib", "logd.joblib")
GENERATOR_FILES = ("generator.pt", "train_smiles.txt")
MIN_FREE_GB = 2.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str | None = None  # command that resolves it
    optional: bool = False


def run_checks(data_dir: Path = DATA_DIR, model_dir: Path = MODEL_DIR) -> list[Check]:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    checks = [Check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0],
                    "install Python 3.10 or newer")]
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    checks.append(Check("Python packages", not missing,
                        "all installed" if not missing else f"missing: {', '.join(missing)}",
                        "pip install -r requirements.txt"))
    absent = [f for f in DATA_FILES if not (data_dir / f).exists()]
    checks.append(Check("Training data", not absent,
                        "ZINC250k, ESOL, Lipophilicity present" if not absent else f"missing: {', '.join(absent)}",
                        "python -m chemgen download"))
    absent = [f for f in PREDICTOR_FILES if not (model_dir / f).exists()]
    checks.append(Check("Property predictors", not absent,
                        "solubility and logD models trained" if not absent else f"missing: {', '.join(absent)}",
                        "python -m chemgen train-predictors"))
    absent = [f for f in GENERATOR_FILES if not (model_dir / f).exists()]
    checks.append(Check("Molecule generator", not absent,
                        "trained" if not absent else f"missing: {', '.join(absent)}",
                        "python -m chemgen train-generator"))
    free_gb = shutil.disk_usage(data_dir if data_dir.exists() else Path(".")).free / 1e9
    checks.append(Check("Free disk space", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free (need {MIN_FREE_GB:g})",
                        "free some disk space"))
    harvested = data_dir / "harvested.csv"
    rows = max(0, sum(1 for _ in harvested.open()) - 1) if harvested.exists() else 0
    checks.append(Check("Harvested molecules (optional)", rows > 0,
                        f"{rows} molecules in harvested.csv" if rows else "none yet",
                        "python -m chemgen harvest --budget 5000", optional=True))
    return checks


def render(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "ok  " if c.ok else ("skip" if c.optional else "FAIL")
        lines.append(f"[{mark}] {c.name}: {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"       -> {c.fix}")
    ready = all(c.ok for c in checks if not c.optional)
    lines.append("")
    lines.append("Ready: run `streamlit run app/streamlit_app.py`" if ready
                 else "Not ready yet: run `python -m chemgen prepare` to do the missing steps.")
    return "\n".join(lines)


def is_ready(checks: list[Check]) -> bool:
    return all(c.ok for c in checks if not c.optional)


def prepare(quick: bool = False, harvest_budget: int = 0, data_dir: Path = DATA_DIR,
            model_dir: Path = MODEL_DIR, log: Callable[[str], None] = print,
            runners: dict[str, Callable[[], None]] | None = None) -> list[str]:
    """Run whatever is missing, in order. ``quick`` trains a smaller generator (about 10 minutes).

    ``runners`` lets tests replace the expensive steps. Returns the names of the steps it ran.
    """
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def download() -> None:
        from .data import download as fetch
        fetch(data_dir, log=log)

    def train_predictors() -> None:
        import json
        from .predictor import DATASETS, PropertyPredictor
        report = {}
        for name in DATASETS:
            predictor, metrics = PropertyPredictor.train(name, data_dir)
            predictor.save(model_dir / f"{name}.joblib")
            report[name] = metrics
            log(f"  {name}: scaffold-split RMSE {metrics['rmse']:.3f} (baseline {metrics['baseline_rmse']:.3f})")
        (model_dir / "predictor_metrics.json").write_text(json.dumps(report, indent=2))

    def train_generator() -> None:
        import json
        from .generator import train_generator as train
        extra = data_dir / "harvested.csv"
        result = train(data_dir / "zinc250k.csv", model_dir / "generator.pt", model_dir / "train_smiles.txt",
                       n_mols=20_000 if quick else 60_000, epochs=5 if quick else 10, log=log,
                       extra_csvs=[extra] if extra.exists() else [])
        (model_dir / "generator_metrics.json").write_text(json.dumps(result, indent=2))

    def harvest() -> None:
        from .harvest import harvest as run
        run(["pubchem", "chembl"], harvest_budget, data_dir / "harvested.csv", cache_dir=data_dir / "cache", log=log)

    runners = runners or {}
    steps = [("download", lambda c: not c["Training data"].ok, download),
             ("train-predictors", lambda c: not c["Property predictors"].ok, train_predictors),
             ("harvest", lambda c: harvest_budget > 0, harvest),
             ("train-generator", lambda c: not c["Molecule generator"].ok, train_generator)]
    done = []
    for name, needed, action in steps:
        checks = {c.name: c for c in run_checks(data_dir, model_dir)}
        if needed(checks):
            log(f"== {name}")
            runners.get(name, action)()
            done.append(name)
    return done
