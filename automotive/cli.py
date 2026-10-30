"""Command line interface: python -m automotive {download,train-predictors,train-generator,
design,apply,status,prepare}."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import DATA_DIR, MODEL_DIR


def _download(args) -> None:
    from .data import download
    download(Path(args.data_dir), force=args.force)


def _train_predictors(args) -> None:
    import json
    from .predictor import DATASETS, train
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in DATASETS:
        predictor, metrics = train(name, Path(args.data_dir))
        predictor.save(MODEL_DIR / f"{name}.joblib")
        report[name] = metrics
        print(f"{name}: scaffold-split RMSE {metrics['rmse']:.3f} (mean-baseline {metrics['baseline_rmse']:.3f}), "
              f"R2 {metrics['r2']:.3f}, n={metrics['n_molecules']}")
    (MODEL_DIR / "predictor_metrics.json").write_text(json.dumps(report, indent=2))


def _train_generator(args) -> None:
    import json
    from .generator import train_generator
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    result = train_generator(Path(args.data_dir), MODEL_DIR / "generator.pt", MODEL_DIR / "train_smiles.txt",
                             Path(args.data_dir) / "generator_train.csv", n_mols=args.n_mols, epochs=args.epochs)
    (MODEL_DIR / "generator_metrics.json").write_text(json.dumps(result, indent=2))


def _design(args) -> None:
    import pandas as pd
    from .pipeline import AutomotivePipeline, AutomotiveTargets
    top_k = args.top
    targets = AutomotiveTargets(logp=args.logp, tpsa=args.tpsa, sas=args.sas, role=args.role)
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
    pipeline = AutomotivePipeline(MODEL_DIR, Path(args.data_dir))
    result = pipeline.design(targets, n_samples=args.n, top_k=top_k, temperature=args.temperature,
                             seed=args.seed, max_steps=args.steps)
    print(" -> ".join(f"{k} {v}" for k, v in result.stats.items()))
    if result.table.empty:
        print("no molecules passed the filters; try more samples or a lower temperature")
        return
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(result.table[["SMILES", "score", "logP", "TPSA", "SAS", "pred_bp", "pred_vp", "pred_mp",
                            "pred_ws", "role_tags"]].round(2).to_string())
    if args.out:
        result.table.to_csv(args.out, index=False)
        print(f"saved {args.out}")
    if args.recipes_out:
        Path(args.recipes_out).write_text(result.recipes_markdown())
        print(f"saved synthesis recipes to {args.recipes_out}")
    for smiles in list(result.table["SMILES"])[:args.recipes]:
        recipe = result.recipes.get(smiles)
        print(f"\n{'=' * 78}\n{smiles}\n{'=' * 78}")
        print(recipe.to_markdown() if recipe else "No route from the supported reaction classes within the step limit.")


def _apply(args) -> None:
    from .apply import ModelBundle, analyze
    bundle = ModelBundle.load(MODEL_DIR, Path(args.data_dir))
    text = analyze(args.smiles, bundle, scale_mmol=args.scale, max_steps=args.steps).to_markdown()
    if args.out:
        Path(args.out).write_text(text)
        print(f"saved {args.out}")
    else:
        print(text)


def _status(args) -> None:
    from .prepare import is_ready, render, run_checks
    checks = run_checks(Path(args.data_dir), MODEL_DIR)
    print(render(checks))
    if not is_ready(checks):
        raise SystemExit(1)


def _prepare(args) -> None:
    from .prepare import prepare, render, run_checks
    done = prepare(quick=args.quick, data_dir=Path(args.data_dir), model_dir=MODEL_DIR)
    print(f"\nsteps run: {', '.join(done) or 'none (already prepared)'}\n")
    print(render(run_checks(Path(args.data_dir), MODEL_DIR)))


def main() -> None:
    parser = argparse.ArgumentParser(prog="automotive")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the OPERA physical-property data")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_download)

    p = sub.add_parser("train-predictors", help="train the boiling point / vapor pressure / melting "
                                                "point / water solubility predictors")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_train_predictors)

    p = sub.add_parser("train-generator", help="train the conditional SMILES generator")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--n-mols", type=int, default=20_000)
    p.add_argument("--epochs", type=int, default=10)
    p.set_defaults(func=_train_generator)

    p = sub.add_parser("design", help="generate and rank automotive chemicals for target properties")
    p.add_argument("--logp", type=float, default=4.0)
    p.add_argument("--tpsa", type=float, default=20.0)
    p.add_argument("--sas", type=float, default=3.0)
    p.add_argument("--role", choices=["corrosion_inhibitor", "antiwear_ep", "antioxidant",
                                      "friction_modifier", "base_oil_like", "fuel_range_hydrocarbon"])
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--english", help='describe what you need in words, e.g. "a nonpolar base-oil-like '
                                     'hydrocarbon, easy to make, show 10"')
    p.add_argument("--recipes", type=int, default=0, metavar="N")
    p.add_argument("--recipes-out")
    p.add_argument("--out")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_design)

    p = sub.add_parser("apply", help="analyse a structure: screen, predict, tag its automotive role, "
                                     "how to prepare it")
    p.add_argument("--smiles", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--out")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_apply)

    p = sub.add_parser("status", help="check that everything the app needs is installed and trained")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_status)

    p = sub.add_parser("prepare", help="download and train whatever is missing")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.set_defaults(func=_prepare)

    args = parser.parse_args()
    args.func(args)
