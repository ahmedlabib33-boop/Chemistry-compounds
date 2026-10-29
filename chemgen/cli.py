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
                             MODEL_DIR / "train_smiles.txt", n_mols=args.n_mols, epochs=args.epochs,
                             extra_csvs=[Path(p) for p in args.extra])
    (MODEL_DIR / "generator_metrics.json").write_text(json.dumps(result, indent=2))


def _harvest(args) -> None:
    from .harvest import SOURCES, harvest
    unknown = set(args.sources) - set(SOURCES)
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(sorted(unknown))}; choose from {', '.join(SOURCES)}")
    data_dir = Path(args.data_dir)
    harvest(args.sources, args.budget, Path(args.out) if args.out else data_dir / "harvested.csv",
            cache_dir=data_dir / "cache", seed=args.seed)


def _lookup(args) -> None:
    from .harvest import HttpClient, Refused, lookup
    kind, value = next((k, v) for k, v in
                       (("formula", args.formula), ("name", args.name), ("similar", args.similar)) if v)
    try:
        result = lookup(kind, value, limit=args.max, threshold=args.threshold,
                        client=HttpClient(cache_dir=Path(args.data_dir) / "cache"))
    except Refused as err:
        print(err)
        return
    except ValueError as err:
        raise SystemExit(str(err))
    print(f"PubChem {kind} '{value}': {len(result.accepted)} structures pass ChemGen's screens"
          + (f"; filtered out: {dict(result.rejected)}" if result.rejected else ""))
    for row in result.accepted:
        print(f"  CID {row['source_id']:>10}  {row['formula'] or '':14} {row['smiles']}   "
              f"logP {row['logP']:.2f}  QED {row['qed']:.2f}  SAS {row['SAS']:.2f}")


def _design(args) -> None:
    import pandas as pd
    from .pipeline import DesignPipeline, DesignTargets
    top_k = args.top
    targets = DesignTargets(logp=args.logp, qed=args.qed, sas=args.sas, min_logs=args.min_logs)
    if args.english:
        from .nlu import parse_request
        request = parse_request(args.english)
        if request.refusal:
            print(request.refusal)
            return
        targets, top_k = request.targets, request.top_k or top_k
        print("Understood:")
        for line in request.interpretation:
            print(f"  - {line}")
        for line in request.unsupported:
            print(f"  ! not modelled: {line}")
    pipeline = DesignPipeline(MODEL_DIR)
    result = pipeline.design(targets, n_samples=args.n, top_k=top_k,
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
    if args.recipes_out:
        Path(args.recipes_out).write_text(result.recipes_markdown())
        print(f"saved synthesis recipes to {args.recipes_out}")
    for smiles in list(result.table["SMILES"])[:args.recipes]:
        recipe = result.recipes.get(smiles)
        print(f"\n{'=' * 78}\n{smiles}\n{'=' * 78}")
        print(recipe.to_markdown() if recipe else "No one-step route from the supported reaction classes.")


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
    p.add_argument("--extra", nargs="*", default=[], metavar="CSV",
                   help="harvested CSV files to mix into the training data (see 'harvest')")
    p.set_defaults(func=_train_generator)

    p = sub.add_parser("harvest", help="agents that pull molecules from PubChem / ChEMBL")
    p.add_argument("--sources", nargs="+", default=["pubchem", "chembl"])
    p.add_argument("--budget", type=int, default=2000, help="new molecules to accept per source")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--out", help="output CSV (default: data/harvested.csv)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_harvest)

    p = sub.add_parser("lookup", help="look a formula, name or structure up on PubChem")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--formula", help="molecular formula, e.g. C9H8O4")
    group.add_argument("--name", help="compound name, e.g. ibuprofen")
    group.add_argument("--similar", metavar="SMILES", help="structures similar to this SMILES")
    p.add_argument("--max", type=int, default=25)
    p.add_argument("--threshold", type=int, default=90, help="similarity threshold (percent)")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_lookup)

    p = sub.add_parser("design", help="generate and rank molecules for target properties")
    p.add_argument("--logp", type=float, default=2.5, help="target logP")
    p.add_argument("--qed", type=float, default=0.8, help="minimum QED drug-likeness")
    p.add_argument("--sas", type=float, default=3.0, help="maximum synthetic accessibility score")
    p.add_argument("--min-logs", type=float, default=-4.0, help="minimum predicted logS")
    p.add_argument("--n", type=int, default=2000, help="molecules to sample")
    p.add_argument("--top", type=int, default=20, help="molecules to keep")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, help="make sampling reproducible")
    p.add_argument("--english", help='describe the molecule in words, e.g. "water soluble, easy to make, '
                                     'drug-like, logP around 3" (overrides the numeric options)')
    p.add_argument("--recipes", type=int, default=0, metavar="N",
                   help="print the synthesis recipe for the top N molecules")
    p.add_argument("--recipes-out", help="write every recipe to this Markdown file")
    p.add_argument("--out", help="write the result table to this CSV")
    p.set_defaults(func=_design)

    args = parser.parse_args()
    args.func(args)
