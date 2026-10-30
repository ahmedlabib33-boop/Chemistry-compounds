"""Readiness check and one-command preparation, mirroring chemgen.prepare for this domain."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from . import DATA_DIR, MODEL_DIR

REQUIRED_PACKAGES = {"rdkit": "rdkit", "torch": "torch", "sklearn": "scikit-learn", "pandas": "pandas",
                     "numpy": "numpy", "joblib": "joblib", "streamlit": "streamlit"}
DATA_FILES = ("bp.csv", "vp.csv", "mp.csv", "ws.csv")
PREDICTOR_FILES = ("bp.joblib", "vp.joblib", "mp.joblib", "ws.joblib")
GENERATOR_FILES = ("generator.pt", "train_smiles.txt")
MIN_FREE_GB = 1.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str | None = None


def run_checks(data_dir: Path = DATA_DIR, model_dir: Path = MODEL_DIR) -> list[Check]:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    checks = [Check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0],
                    "install Python 3.10 or newer")]
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items() if importlib.util.find_spec(mod) is None]
    checks.append(Check("Python packages", not missing,
                        "all installed" if not missing else f"missing: {', '.join(missing)}",
                        "pip install -r requirements.txt"))
    absent = [f for f in DATA_FILES if not (data_dir / f).exists()]
    checks.append(Check("OPERA property data", not absent,
                        "boiling point, vapor pressure, melting point, water solubility present"
                        if not absent else f"missing: {', '.join(absent)}", "python -m automotive download"))
    absent = [f for f in PREDICTOR_FILES if not (model_dir / f).exists()]
    checks.append(Check("Property predictors", not absent,
                        "all four trained" if not absent else f"missing: {', '.join(absent)}",
                        "python -m automotive train-predictors"))
    absent = [f for f in GENERATOR_FILES if not (model_dir / f).exists()]
    checks.append(Check("Molecule generator", not absent,
                        "trained" if not absent else f"missing: {', '.join(absent)}",
                        "python -m automotive train-generator"))
    free_gb = shutil.disk_usage(data_dir if data_dir.exists() else Path(".")).free / 1e9
    checks.append(Check("Free disk space", free_gb >= MIN_FREE_GB, f"{free_gb:.1f} GB free (need {MIN_FREE_GB:g})",
                        "free some disk space"))
    return checks


def render(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{'ok  ' if c.ok else 'FAIL'}] {c.name}: {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"       -> {c.fix}")
    ready = all(c.ok for c in checks)
    lines.append("")
    lines.append("Ready: run `streamlit run app/streamlit_app.py` and open the Automotive page" if ready
                 else "Not ready yet: run `python -m automotive prepare` to do the missing steps.")
    return "\n".join(lines)


def is_ready(checks: list[Check]) -> bool:
    return all(c.ok for c in checks)


def prepare(quick: bool = False, data_dir: Path = DATA_DIR, model_dir: Path = MODEL_DIR,
            log=print, runners: dict | None = None) -> list[str]:
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def download() -> None:
        from .data import download as fetch
        fetch(data_dir, log=log)

    def train_predictors() -> None:
        import json
        from .predictor import DATASETS, train
        report = {}
        for name in DATASETS:
            predictor, metrics = train(name, data_dir)
            predictor.save(model_dir / f"{name}.joblib")
            report[name] = metrics
            log(f"  {name}: scaffold-split RMSE {metrics['rmse']:.3f} (baseline {metrics['baseline_rmse']:.3f})")
        (model_dir / "predictor_metrics.json").write_text(json.dumps(report, indent=2))

    def train_generator() -> None:
        import json
        from .generator import train_generator as train
        result = train(data_dir, model_dir / "generator.pt", model_dir / "train_smiles.txt",
                       data_dir / "generator_train.csv", n_mols=8_000 if quick else 20_000,
                       epochs=5 if quick else 10, log=log)
        (model_dir / "generator_metrics.json").write_text(json.dumps(result, indent=2))

    runners = runners or {}
    steps = [("download", lambda c: not c["OPERA property data"].ok, download),
             ("train-predictors", lambda c: not c["Property predictors"].ok, train_predictors),
             ("train-generator", lambda c: not c["Molecule generator"].ok, train_generator)]
    done = []
    for name, needed, action in steps:
        checks = {c.name: c for c in run_checks(data_dir, model_dir)}
        if needed(checks):
            log(f"== {name}")
            runners.get(name, action)()
            done.append(name)
    return done
