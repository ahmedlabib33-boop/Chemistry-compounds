# ChemGen

Property-conditioned generation of drug-like molecules, with property prediction, safety
filtering and ranking. You choose target properties (lipophilicity, drug-likeness,
synthetic accessibility, solubility); the app proposes novel molecules that meet them.

```
target properties
      |
      v
 conditional SMILES generator (GRU)  -->  validity check  -->  dedupe  -->  novelty vs training set
      |
      v
 safety filters (elements, fragments, MW, Lipinski, PAINS/NIH alerts)
      |
      v
 property models (solubility, logD; random forest + tree-spread uncertainty)
      |
      v
 desirability score  -->  diversity-aware top-k  -->  UI / CSV
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m chemgen download                 # ZINC250k, ESOL, Lipophilicity (open data)
python -m chemgen train-predictors         # ~2 min on CPU
python -m chemgen train-generator          # ~35 min on 4 CPU cores (60k molecules, 10 epochs)

python -m chemgen design --logp 2.5 --qed 0.8 --sas 3 --min-logs -4 --n 2000 --top 20 --out hits.csv
streamlit run app/streamlit_app.py
```

`train-generator` accepts `--n-mols` (up to the full 250k) and `--epochs`; more data and
epochs raise the fraction of valid, novel molecules.

## Components

| Module | Role |
| --- | --- |
| `chemgen/generator.py` | GRU language model over SMILES tokens. The condition vector (logP, QED, SA score) sets the initial hidden state and is fed at every step. Trained on ZINC250k. |
| `chemgen/predictor.py` | Random forests on Morgan count fingerprints plus all RDKit 2D descriptors. Trained on ESOL (solubility) and Lipophilicity (logD 7.4). Evaluated on a scaffold split. Uncertainty is the standard deviation across trees. |
| `chemgen/safety.py` | Hard filters on every generated molecule. See below. |
| `chemgen/pipeline.py` | The sampling, filtering, scoring and ranking loop. |
| `app/streamlit_app.py` | Web UI with a molecule grid, metrics, table and CSV download. |

## Safety filters

Every molecule must pass all of these before it is scored or shown:

- only C, N, O, S, F, Cl, Br, I (plus H); no metals, phosphorus, boron, silicon, etc.
- a single connected fragment, no radicals
- molecular weight 150-600 and at most one Lipinski violation
- no PAINS or NIH structural alerts (assay-interfering, reactive and toxic motifs)

Brenk alerts (unwanted functionality) do not reject a molecule but reduce its score.
The generator is trained only on drug-like ZINC molecules and these filters apply on top
of that. The system is scoped to drug-like small molecules; it does not provide
synthesis routes.

## Honest limitations

- The solubility and logD models are trained on about 1,100 and 4,200 molecules. Treat
  their output as screening estimates, and use the `*_std` columns to spot molecules far
  from the training data.
- logP, QED and SA score are computed with RDKit's empirical models, not measured.
- Generated molecules are computational proposals. Nothing here predicts bioactivity,
  toxicity, ADMET or whether a compound can actually be made. Those need experimental
  work or purpose-built models.

## Tests

```bash
python -m pytest tests -q
```
