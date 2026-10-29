"""Command line interface: python -m chemgen {download,train-predictors,train-generator,design}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import DATA_DIR, MODEL_DIR


def _download(args) -> None:
    from .data import download
    download(Path(args.data_dir), force=args.force)


def _train_predictors(args) -> None:
    from .predictor import DATASETS, PropertyPredictor
    MODEL_DIR.mkdir(exist_ok=True)
    report = {}
    for name in DATASETS:
        predictor, metrics = PropertyPredictor.train(name, Path(args.data_dir))
        predictor.save(MODEL_DIR / f"{name}.joblib")
        report[name] = metrics
        print(f"{name}: scaffold-split RMSE {metrics['rmse']:.3f} "
              f"(mean-baseline {metrics['baseline_rmse']:.3f}), R2 {metrics['r2']:.3f}")
    (MODEL_DIR / "predictor_metrics.json").write_text(json.dumps(report, indent=2))


def _train_generator(args) -> None:
    from .generator import train_generator
    MODEL_DIR.mkdir(exist_ok=True)
    result = train_generator(Path(args.data_dir) / "zinc250k.csv", MODEL_DIR / "generator.pt",
                             MODEL_DIR / "train_smiles.txt", n_mols=args.n_mols, epochs=args.epochs)
    (MODEL_DIR / "generator_metrics.json").write_text(json.dumps(result, indent=2))


def _design(args) -> None:
    import pandas as pd
    from .pipeline import DesignPipeline, DesignTargets
    pipeline = DesignPipeline(MODEL_DIR)
    targets = DesignTargets(logp=args.logp, qed=args.qed, sas=args.sas, min_logs=args.min_logs)
    result = pipeline.design(targets, n_samples=args.n, top_k=args.top,
                             temperature=args.temperature, seed=args.seed)
    print(" -> ".join(f"{k} {v}" for k, v in result.stats.items()))
    if result.table.empty:
        print("no molecules passed the filters; try more samples or a lower temperature")
        return
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(result.table[["SMILES", "score", "logP", "QED", "SAS", "pred_logS"]]
              .round(2).to_string())
    if args.out:
        result.table.to_csv(args.out, index=False)
        print(f"saved {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="chemgen")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the open training datasets")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_download)

    p = sub.add_parser("train-predictors", help="train solubility and logD predictors")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_train_predictors)

    p = sub.add_parser("train-generator", help="train the conditional SMILES generator")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--n-mols", type=int, default=100_000)
    p.add_argument("--epochs", type=int, default=12)
    p.set_defaults(func=_train_generator)

    p = sub.add_parser("design", help="generate and rank molecules for target properties")
    p.add_argument("--logp", type=float, default=2.5, help="target logP")
    p.add_argument("--qed", type=float, default=0.8, help="minimum QED drug-likeness")
    p.add_argument("--sas", type=float, default=3.0, help="maximum synthetic accessibility score")
    p.add_argument("--min-logs", type=float, default=-4.0, help="minimum predicted logS")
    p.add_argument("--n", type=int, default=2000, help="molecules to sample")
    p.add_argument("--top", type=int, default=20, help="molecules to keep")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, help="make sampling reproducible")
    p.add_argument("--out", help="write the result table to this CSV")
    p.set_defaults(func=_design)

    args = parser.parse_args()
    args.func(args)
