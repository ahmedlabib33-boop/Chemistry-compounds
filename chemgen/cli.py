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


def _apply(args) -> None:
    from .apply import ModelBundle, analyze
    from .harvest import HttpClient, Refused, candidates
    bundle = ModelBundle.load()
    if args.smiles:
        smiles_list = [args.smiles]
    else:
        kind, value = ("formula", args.formula) if args.formula else ("name", args.name)
        try:
            records = candidates(kind, value, limit=args.max,
                                 client=HttpClient(cache_dir=Path(args.data_dir) / "cache"))
        except Refused as err:
            print(err)
            return
        except ValueError as err:
            raise SystemExit(str(err))
        print(f"PubChem {kind} '{value}': {len(records)} structure(s) found")
        smiles_list = [r.smiles for r in records[:args.max]]
    reports = [analyze(s, bundle, scale_mmol=args.scale).to_markdown() for s in smiles_list]
    text = "\n\n---\n\n".join(reports)
    if args.out:
        Path(args.out).write_text(text)
        print(f"saved {len(reports)} report(s) to {args.out}")
    else:
        print(text)


def _automate(args) -> None:
    from .automation import CycleConfig, run_cycle, run_forever
    cfg = CycleConfig(sources=tuple(args.sources), harvest_budget=args.harvest_budget, min_new=args.min_new,
                      n_mols=args.n_mols, epochs=args.epochs, eval_n=args.eval_n, min_gain=args.min_gain,
                      seed=args.seed, skip_harvest=args.skip_harvest, force=args.force,
                      promote=not args.no_promote, data_dir=Path(args.data_dir))
    if args.loop:
        print(f"running a cycle every {args.interval_hours:g} h; stop with Ctrl-C")
        run_forever(cfg, args.interval_hours)
    else:
        run_cycle(cfg)


def _rollback(args) -> None:
    from .automation import Registry
    try:
        print(f"live model is now {Registry(MODEL_DIR).rollback()}")
    except RuntimeError as err:
        raise SystemExit(str(err))


def _status(args) -> None:
    from .prepare import is_ready, render, run_checks
    checks = run_checks(Path(args.data_dir), MODEL_DIR)
    print(render(checks))
    if not is_ready(checks):
        raise SystemExit(1)


def _prepare(args) -> None:
    from .prepare import is_ready, prepare, render, run_checks
    done = prepare(quick=args.quick, harvest_budget=args.harvest, data_dir=Path(args.data_dir),
                   model_dir=MODEL_DIR)
    print(f"\nsteps run: {', '.join(done) or 'none (already prepared)'}\n")
    print(render(run_checks(Path(args.data_dir), MODEL_DIR)))


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

    p = sub.add_parser("apply", help="analyse a structure, formula or name: screen, predict, how to prepare it")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", help="a structure, e.g. CC(=O)Nc1ccc(O)cc1")
    group.add_argument("--formula", help="a molecular formula, e.g. C9H8O4 (looks isomers up on PubChem)")
    group.add_argument("--name", help="a compound name, e.g. ibuprofen (looked up on PubChem)")
    p.add_argument("--max", type=int, default=5, help="most structures to analyse for a formula or name")
    p.add_argument("--scale", type=float, default=1.0, help="reaction scale in mmol for the recipe")
    p.add_argument("--out", help="write the report(s) to this Markdown file")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_apply)

    p = sub.add_parser("automate", help="harvest -> retrain -> benchmark -> promote if better")
    p.add_argument("--sources", nargs="+", default=["pubchem", "chembl"])
    p.add_argument("--harvest-budget", type=int, default=5000, help="new molecules to accept per source")
    p.add_argument("--min-new", type=int, default=500, help="retrain only if this many new molecules arrived")
    p.add_argument("--n-mols", type=int, default=60_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=400, help="samples per benchmark condition")
    p.add_argument("--min-gain", type=float, default=0.005, help="useful-yield gain needed to promote")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-harvest", action="store_true")
    p.add_argument("--force", action="store_true", help="retrain even if little new data arrived")
    p.add_argument("--no-promote", action="store_true", help="evaluate only; never replace the live model")
    p.add_argument("--loop", action="store_true", help="repeat forever instead of running once")
    p.add_argument("--interval-hours", type=float, default=168.0)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_automate)

    p = sub.add_parser("rollback", help="restore the previously live generator")
    p.set_defaults(func=_rollback)

    p = sub.add_parser("status", help="check that everything the app needs is installed and trained")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_status)

    p = sub.add_parser("prepare", help="download, train and (optionally) harvest whatever is missing")
    p.add_argument("--quick", action="store_true", help="smaller generator (about 10 minutes)")
    p.add_argument("--harvest", type=int, default=0, metavar="N", help="also harvest N molecules per source")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_prepare)

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
