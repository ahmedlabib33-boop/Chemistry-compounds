"""Closed-loop ML automation: harvest new data, retrain a candidate, benchmark it against the
live model, and promote it only if it is better. Every promotion keeps the old model in a
versioned registry so it can be rolled back.

One cycle:
    1. harvest      pull new molecules from PubChem / ChEMBL (curated and screened)
    2. gate         skip retraining unless enough new data has arrived since the last training
    3. train        fit a candidate generator on ZINC + all harvested molecules
    4. evaluate     sample both candidate and live model on the same conditions and seeds
    5. promote      swap the candidate in iff its useful-molecule yield is clearly higher
    6. log          append the full cycle record to models/automation_log.jsonl

"Useful" means a sampled molecule that is valid, a first occurrence, not in either model's
training set, passes every safety screen, and actually lands near the requested logP / QED / SA.
The external side effects (network, training) are injected so the loop is testable.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
from rdkit import Chem

from . import DATA_DIR, MODEL_DIR, chem, safety

LIVE_FILES = ("generator.pt", "train_smiles.txt")
META_FILE = "model_meta.json"
LOG_FILE = "automation_log.jsonl"

# (target logP, target QED, target SA): the benchmark grid both models are scored on
DEFAULT_CONDITIONS = ((1.0, 0.7, 2.5), (2.5, 0.8, 3.0), (3.5, 0.7, 3.0), (2.0, 0.85, 2.5))
LOGP_TOLERANCE, QED_SLACK, SAS_SLACK = 0.75, 0.1, 0.5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def read_meta(model_dir: Path) -> dict:
    path = Path(model_dir) / META_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def _replace_atomically(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


class Registry:
    """Live model files sit in ``model_dir``; every superseded version is archived beside them."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.root = self.model_dir / "registry"

    def new_label(self) -> str:
        """A timestamp label that differs from every archived version and from the live one."""
        base = _now().strftime("%Y%m%d-%H%M%S")
        taken = set(self.versions()) | {self.live_version()}
        label, n = base, 1
        while label in taken:
            n += 1
            label = f"{base}-{n}"
        return label

    def versions(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.exists() else []

    def live_version(self) -> str | None:
        if not all((self.model_dir / f).exists() for f in LIVE_FILES):
            return None
        return read_meta(self.model_dir).get("version", "initial")

    def _archive_live(self) -> str | None:
        label = self.live_version()
        if label is None:
            return None
        target = self.root / label
        if not target.exists():
            target.mkdir(parents=True)
            for name in (*LIVE_FILES, META_FILE):
                if (self.model_dir / name).exists():
                    shutil.copy2(self.model_dir / name, target / name)
        return label

    def _install(self, source: Path, meta: dict) -> None:
        for name in LIVE_FILES:
            _replace_atomically(Path(source) / name, self.model_dir / name)
        tmp = self.model_dir / (META_FILE + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(self.model_dir / META_FILE)

    def promote(self, candidate_dir: Path, meta: dict) -> str:
        """Archive the live model, then install the candidate as the new live version."""
        previous = self._archive_live()
        version = meta.get("version") or self.new_label()
        self._install(candidate_dir, {**meta, "version": version, "previous_version": previous,
                                      "promoted_at": _now().isoformat()})
        return version

    def rollback(self) -> str:
        """Re-install the version that was live before this one. Repeat to keep walking back."""
        current = self._archive_live()
        target = read_meta(self.model_dir).get("previous_version") if current else None
        if not target or not (self.root / target).exists():
            raise RuntimeError("no earlier version to roll back to")
        meta = read_meta(self.root / target) or {}
        self._install(self.root / target, {**meta, "version": target, "rolled_back_from": current})
        return target


# ---------- evaluation ----------

def evaluate_generator(generator, known: set[str], conditions=DEFAULT_CONDITIONS, n: int = 400,
                       seed: int = 0) -> dict:
    """Sample ``n`` molecules per condition and measure how many are actually useful."""
    torch.manual_seed(seed)
    totals = dict(sampled=0, valid=0, unique=0, novel=0, safe=0, useful=0)
    for logp, qed, sas in conditions:
        seen: set[str] = set()
        for smiles in generator.sample(logp, qed, sas, n=n):
            totals["sampled"] += 1
            mol = chem.to_mol(smiles)
            if mol is None:
                continue
            totals["valid"] += 1
            mol = chem.neutralize(mol)
            canonical = Chem.MolToSmiles(mol)
            if canonical in seen:
                continue
            seen.add(canonical)
            totals["unique"] += 1
            if canonical in known:
                continue
            totals["novel"] += 1
            if not safety.assess(mol).passed:
                continue
            totals["safe"] += 1
            props = chem.basic_properties(mol)
            if (abs(props["logP"] - logp) <= LOGP_TOLERANCE and props["QED"] >= qed - QED_SLACK
                    and props["SAS"] <= sas + SAS_SLACK):
                totals["useful"] += 1
    sampled = max(totals["sampled"], 1)
    return {**totals, "validity": totals["valid"] / sampled,
            "useful_yield": totals["useful"] / sampled, "conditions": [list(c) for c in conditions],
            "n_per_condition": n, "seed": seed}


def decide(candidate: dict, current: dict | None, min_gain: float = 0.005,
           max_validity_drop: float = 0.02) -> tuple[bool, str]:
    if current is None:
        return True, "no live model yet"
    gain = candidate["useful_yield"] - current["useful_yield"]
    drop = current["validity"] - candidate["validity"]
    if drop > max_validity_drop:
        return False, f"validity fell {drop:.1%} (limit {max_validity_drop:.1%})"
    if gain < min_gain:
        return False, (f"useful yield {candidate['useful_yield']:.1%} vs live {current['useful_yield']:.1%}: "
                       f"gain {gain:+.1%} is below the required {min_gain:.1%}")
    return True, f"useful yield {current['useful_yield']:.1%} -> {candidate['useful_yield']:.1%}"


# ---------- the cycle ----------

@dataclass
class CycleConfig:
    sources: tuple[str, ...] = ("pubchem", "chembl")
    harvest_budget: int = 5000
    min_new: int = 500
    n_mols: int = 60_000
    epochs: int = 10
    eval_n: int = 400
    min_gain: float = 0.005
    seed: int = 0
    skip_harvest: bool = False
    force: bool = False
    promote: bool = True
    data_dir: Path = DATA_DIR
    model_dir: Path = MODEL_DIR


@dataclass
class CycleReport:
    started: str
    steps: list[dict] = field(default_factory=list)
    harvested_total: int = 0
    new_since_training: int = 0
    retrained: bool = False
    candidate_metrics: dict | None = None
    live_metrics: dict | None = None
    promoted: bool = False
    decision: str = ""
    live_version: str | None = None

    def step(self, name: str, status: str, detail: str = "") -> None:
        self.steps.append({"step": name, "status": status, "detail": detail})

    def summary(self) -> str:
        lines = [f"cycle started {self.started}"]
        lines += [f"  {s['step']:<9} {s['status']:<8} {s['detail']}" for s in self.steps]
        lines.append(f"  => {'PROMOTED' if self.promoted else 'live model unchanged'}: {self.decision}")
        return "\n".join(lines)


def count_rows(path: Path) -> int:
    path = Path(path)
    return max(0, sum(1 for _ in path.open()) - 1) if path.exists() else 0


def default_harvest(cfg: CycleConfig) -> None:
    from .harvest import harvest
    harvest(list(cfg.sources), cfg.harvest_budget, Path(cfg.data_dir) / "harvested.csv",
            cache_dir=Path(cfg.data_dir) / "cache", seed=cfg.seed, log=lambda *_: None)


def default_train(cfg: CycleConfig, out_dir: Path) -> dict:
    from .generator import train_generator
    extra = Path(cfg.data_dir) / "harvested.csv"
    return train_generator(Path(cfg.data_dir) / "zinc250k.csv", out_dir / "generator.pt",
                           out_dir / "train_smiles.txt", n_mols=cfg.n_mols, epochs=cfg.epochs,
                           seed=cfg.seed, log=lambda *_: None, extra_csvs=[extra] if extra.exists() else [])


def default_evaluate(model_dir: Path, known: set[str], cfg: CycleConfig) -> dict:
    from .generator import Generator
    return evaluate_generator(Generator.load(Path(model_dir) / "generator.pt"), known,
                              n=cfg.eval_n, seed=cfg.seed)


def _train_set(path: Path) -> set[str]:
    path = Path(path)
    return set(path.read_text().splitlines()) if path.exists() else set()


def run_cycle(cfg: CycleConfig, harvest_fn: Callable = default_harvest, train_fn: Callable = default_train,
              evaluate_fn: Callable = default_evaluate, registry: Registry | None = None,
              log: Callable[[str], None] = print) -> CycleReport:
    registry = registry or Registry(cfg.model_dir)
    model_dir, harvested = Path(cfg.model_dir), Path(cfg.data_dir) / "harvested.csv"
    model_dir.mkdir(parents=True, exist_ok=True)
    report = CycleReport(started=_now().isoformat(timespec="seconds"), live_version=registry.live_version())

    def finish() -> CycleReport:
        with (model_dir / LOG_FILE).open("a") as f:
            f.write(json.dumps(asdict(report)) + "\n")
        log(report.summary())
        return report

    # 1. harvest
    if cfg.skip_harvest:
        report.step("harvest", "skipped", "--skip-harvest")
    else:
        try:
            before = count_rows(harvested)
            harvest_fn(cfg)
            report.step("harvest", "ok", f"+{count_rows(harvested) - before} new molecules")
        except Exception as err:  # a network problem must not stop the loop from using what it has
            report.step("harvest", "failed", f"{type(err).__name__}: {err}")

    # 2. gate
    report.harvested_total = count_rows(harvested)
    report.new_since_training = report.harvested_total - read_meta(model_dir).get("harvested_rows", 0)
    has_live = report.live_version is not None
    if has_live and not cfg.force and report.new_since_training < cfg.min_new:
        report.step("gate", "skipped", f"{report.new_since_training} new molecules since the last "
                                         f"training (need {cfg.min_new})")
        report.decision = "not enough new data to retrain"
        return finish()
    report.step("gate", "ok", f"{report.new_since_training} new molecules since the last training")

    # 3. train a candidate
    (model_dir / "candidates").mkdir(exist_ok=True)
    candidate_dir = Path(tempfile.mkdtemp(prefix=_now().strftime("%Y%m%d-%H%M%S-"), dir=model_dir / "candidates"))
    try:
        train_metrics = train_fn(cfg, candidate_dir)
    except Exception as err:
        report.step("train", "failed", f"{type(err).__name__}: {err}")
        report.decision = "training failed; live model unchanged"
        return finish()
    report.retrained = True
    report.step("train", "ok", f"best validation loss {train_metrics.get('best_val_loss', float('nan')):.3f}")

    # 4. evaluate both on identical conditions, seeds and novelty reference
    known = _train_set(candidate_dir / "train_smiles.txt") | _train_set(model_dir / "train_smiles.txt")
    report.candidate_metrics = evaluate_fn(candidate_dir, known, cfg)
    if has_live:
        report.live_metrics = evaluate_fn(model_dir, known, cfg)
    report.step("evaluate", "ok", f"candidate useful yield {report.candidate_metrics['useful_yield']:.1%}"
                + (f", live {report.live_metrics['useful_yield']:.1%}" if report.live_metrics else ""))

    # 5. promote
    ok, reason = decide(report.candidate_metrics, report.live_metrics, cfg.min_gain)
    report.decision = reason
    if ok and cfg.promote:
        meta = {"version": registry.new_label(), "harvested_rows": report.harvested_total,
                "metrics": report.candidate_metrics, "train": {"epochs": cfg.epochs, "n_mols": cfg.n_mols}}
        report.live_version = registry.promote(candidate_dir, meta)
        report.promoted = True
        report.step("promote", "ok", f"live model is now {report.live_version}")
    else:
        report.step("promote", "skipped", "candidate kept in " + str(candidate_dir) if ok else reason)
    return finish()


def run_forever(cfg: CycleConfig, interval_hours: float, max_cycles: int | None = None,
                sleep: Callable[[float], None] = time.sleep, **kwargs) -> list[CycleReport]:
    reports = []
    while max_cycles is None or len(reports) < max_cycles:
        reports.append(run_cycle(cfg, **kwargs))
        if max_cycles is not None and len(reports) >= max_cycles:
            break
        sleep(interval_hours * 3600)
    return reports


def history(model_dir: Path = MODEL_DIR) -> list[dict]:
    path = Path(model_dir) / LOG_FILE
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
