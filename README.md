# ChemGen

Property-conditioned generation of drug-like molecules, with property prediction, safety
filtering, ranking and a detailed multi-step synthesis recipe for each result. Describe what you need
in plain English (or set sliders) and the app proposes novel molecules that meet it. It runs
entirely locally: no API keys, no paid services.

```
English request  -->  rule-based parser (refuses weapons / toxins / controlled substances)
      |
      v
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
 desirability score  -->  diversity-aware top-k
      |
      v
 multi-step retrosynthesis (9 reaction classes)  -->  step-by-step recipes  -->  UI / CSV / Markdown
```

## How to prepare the app

You need Python 3.10+, about 2 GB of free disk space, and internet access for the one-time
data download. Nothing else: no GPU, no accounts, no API keys.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m chemgen status                   # what is installed / downloaded / trained, and what to run next
python -m chemgen prepare                  # runs every missing step (about 40 min on 4 CPU cores)
python -m chemgen prepare --quick          # smaller generator, about 10 min, for a first look
python -m chemgen prepare --harvest 5000   # also pull 5000 molecules per source before training

streamlit run app/streamlit_app.py         # then open the Design, Apply a formula and Automation pages
```

`prepare` is idempotent: it downloads ZINC250k, ESOL and Lipophilicity, trains the two
property predictors (about 2 min), and trains the generator (about 35 min), skipping anything
already done. `status` exits non-zero when something is missing, so it can gate a deployment.
The same steps are available one at a time:

```bash
python -m chemgen download                 # ZINC250k, ESOL, Lipophilicity (open data)
python -m chemgen train-predictors         # ~2 min on CPU
python -m chemgen train-generator          # ~35 min on 4 CPU cores (60k molecules, 10 epochs)
```

## Quick start

```bash

python -m chemgen design --logp 2.5 --qed 0.8 --sas 3 --min-logs -4 --n 2000 --top 20 --out hits.csv

# the same, described in English, with recipes for the top 3 and all recipes saved to a file
python -m chemgen design --english "easy to make, water soluble, drug-like, logP around 2.5, show 10" \
    --recipes 3 --recipes-out recipes.md

streamlit run app/streamlit_app.py
```

`train-generator` accepts `--n-mols` (up to the full 250k) and `--epochs`; more data and
epochs raise the fraction of valid, novel molecules.

### English input

The parser is rule-based. It reads stated values (`logP 3`, `QED 0.85`, `SAS 2.5`,
`logS -3`, `10 molecules`) and descriptive words (`lipophilic`, `hydrophilic`,
`drug-like`, `easy to make`, `water soluble`, `highly soluble`, ...). Stated values win over
words, values are clamped to the supported range, and unspecified targets use defaults.
It always echoes how it read your request. The generator is conditioned only on logP, QED,
SA score and solubility, so requests about size, specific atoms, rings or a biological
target (for example "a kinase inhibitor") are reported as not modelled rather than
silently ignored.

### Synthesis recipes (multi-step)

Each result gets a planned route of up to `--steps` reactions (default 3, at most 6) built from
nine reaction classes: amide coupling (HATU), ester coupling (EDC/DMAP), urea formation (CDI),
sulfonamide formation, reductive amination, SNAr amination, Suzuki coupling, aryl-ether
alkylation and Boc protection. A complex precursor is disconnected again, recursively, until
every branch ends in a simple starting material, and the cheapest plan wins (one unit per
step, extra for chemoselectivity conflicts, a large penalty for a complex precursor that cannot
be broken down). Every intermediate and starting material passes the hazard screen.

The write-up has: a route overview; a table of starting materials with the amount to weigh out;
and, for **every step**, its inputs (which starting materials, and which earlier step's product),
a reagent table with equivalents and amounts, solvent, temperature and time, numbered procedure
and work-up, purification, expected [M+H]+, hazards and warnings (competing functional groups,
stereochemistry). Quantities are scaled back from the final step: each earlier step is sized so
that, at an assumed 80 % yield per step, it delivers what the next step consumes. That 80 % is a
planning assumption, not a prediction, and the write-up says so.

```bash
python -m chemgen design --english "easy to make, drug-like" --steps 4 --recipes 3
python -m chemgen apply --smiles "O=C(NCc1ccc(F)cc1)c1ccc(CN2CCN(c3ncccn3)CC2)cc1" --steps 3 --check-precursors
```

`--check-precursors` looks every starting material up on PubChem. A known compound is not
necessarily one you can buy, so the report says "check a supplier"; a structure PubChem has never
seen is flagged. Without it, small simple precursors (up to 12 heavy atoms, or up to 18 with a low
SA score) are assumed to be available building blocks, a heuristic and not a catalogue lookup.
On 120 designed molecules, 102 (85 %) get a route within three steps; the rest need chemistry
outside the nine classes (for example free alcohols and ethers with no carbonyl handle, free
primary amines, ketones) or whose only route needs a building block the hazard screen rejects.

### Apply a formula

Give the app something you already have and it applies the trained models to it:

```bash
python -m chemgen apply --smiles "CC(=O)Nc1ccc(O)cc1"
python -m chemgen apply --formula C9H8O4 --max 3     # looks isomers up on PubChem, analyses each
python -m chemgen apply --name ibuprofen --scale 2 --out report.md
```

Each report has the computed properties, ML predictions of solubility and logD with an
uncertainty, an applicability check (Tanimoto similarity to the property models' training
data, so you can see when a prediction is an extrapolation), and the preparation recipe.
Three tiers apply. Structures matching the hazard screen are **refused** with no output.
Structures that cannot be handled (unparseable; elements outside C N O S F Cl Br I) are
reported as **unsupported**. Everything else gets predictions; the recipe is **withheld** if the
structure carries NIH reactive or toxic alerts, and a structure with no one-step route says
that it needs a route designed by a chemist. The same function is on the **Apply a formula** page.

### Automation: the app improves itself

```bash
python -m chemgen automate                      # one cycle
python -m chemgen automate --loop --interval-hours 168
python -m chemgen rollback                      # undo the last promotion
```

One cycle: (1) the harvesting agents pull new molecules; (2) a gate skips retraining unless at
least `--min-new` (500) new molecules arrived since the last training; (3) a candidate generator
is trained on ZINC plus everything harvested; (4) the candidate and the live model are sampled on
the same four property conditions, with the same seed and the same novelty reference; (5) the
candidate is **promoted only if its useful-molecule yield is higher by `--min-gain` (0.5
points) and its validity has not dropped by more than 2 points**; (6) the full record is
appended to `models/automation_log.jsonl`. A "useful" molecule is valid, a first occurrence, not
in either training set, passes every safety screen, and lands near the requested logP, QED and SA
score. Superseded models are archived under `models/registry/`, each live model records its
`previous_version`, and `rollback` walks back one version at a time. A network failure during
harvesting does not stop the cycle; a training failure leaves the live model untouched. The
**Automation** page shows readiness, the live model's metrics, the history, and can start a
cycle in the background; the app reloads the generator by itself when a new one is promoted.

To run it on a schedule, for example every Sunday at 03:00:

```
0 3 * * 0 cd /path/to/Chemistry-compounds && .venv/bin/python -m chemgen automate >> models/automation.out 2>&1
```

The property predictors are not retrained automatically: they learn from small labelled
datasets (ESOL, Lipophilicity) that do not grow by harvesting, so only the generator improves.

### Data-harvesting agents

Agents pull more molecules from free public databases through their official APIs, then
clean and screen them before they can reach any model:

```bash
# two agents (PubChem, ChEMBL) run in parallel; each stops after 5000 new accepted molecules
python -m chemgen harvest --sources pubchem chembl --budget 5000     # -> data/harvested.csv

# look up a molecular formula, a name, or structures similar to a SMILES on PubChem
python -m chemgen lookup --formula C9H8O4
python -m chemgen lookup --name ibuprofen
python -m chemgen lookup --similar "CC(=O)Nc1ccc(O)cc1" --threshold 90

# retrain the generator with the harvested molecules always included in full
python -m chemgen train-generator --extra data/harvested.csv
```

| Agent | Job |
| --- | --- |
| `PubChemSource`, `ChEMBLSource` | Stream raw records from one database. Rate-limited (4 requests/s), retried with back-off, cached on disk under `data/cache`. A network failure ends that agent, not the others. |
| `Curator` | Keeps the largest fragment, neutralises charges, canonicalises, rejects SMILES that are too long or unparseable, and applies every safety screen above. Reports why each record was rejected. |
| `Store` | Thread-safe, deduplicating CSV with the source, source id and fetch date of every molecule. |
| `harvest` | The coordinator: one agent per source, run in parallel, with a per-agent report. |

In a live test, 400 molecules per source were accepted in 44 s; of the 1,096 records fetched,
296 (27 %) were rejected, mostly for structural alerts, Lipinski violations, elements outside
the whitelist, SMILES that were too long, and net charge. `lookup` shows only structures that
pass the same screens and says how many were filtered out. Name and structure queries go
through the same refusal screen as the English parser.

Scope: this reads databases that publish an API; it does not scrape arbitrary web pages,
whose formulas cannot be verified. Another database can be added by writing a class with a
`name` and a `stream()` method and registering it in `SOURCES`. Be aware of the licences:
PubChem data is public domain; ChEMBL is CC BY-SA 3.0, so attribute it and share derived
datasets under the same licence. Harvested molecules are only as good as the databases; the
generator's own predictions and recipes carry the same caveats as everything else here.

## Components

| Module | Role |
| --- | --- |
| `chemgen/generator.py` | GRU language model over SMILES tokens. The condition vector (logP, QED, SA score) sets the initial hidden state and is fed at every step. Trained on ZINC250k. |
| `chemgen/predictor.py` | Random forests on Morgan count fingerprints plus all RDKit 2D descriptors. Trained on ESOL (solubility) and Lipophilicity (logD 7.4). Evaluated on a scaffold split. Uncertainty is the standard deviation across trees. |
| `chemgen/safety.py` | Hard filters on every generated molecule. See below. |
| `chemgen/pipeline.py` | The sampling, filtering, scoring and ranking loop. |
| `chemgen/nlu.py` | English request parser. |
| `chemgen/guard.py` | The request-level refusal screen shared by the English parser and the web agents. |
| `chemgen/apply.py` | Screens a user-supplied structure, predicts its properties with an applicability check, and attaches the recipe. |
| `chemgen/automation.py` | The harvest -> retrain -> benchmark -> promote loop, the versioned model registry and rollback. |
| `chemgen/prepare.py` | `status` readiness checks and the idempotent `prepare` command. |
| `chemgen/harvest.py` | Data-harvesting agents: PubChem and ChEMBL sources, curator, store, coordinator, `lookup`. |
| `chemgen/synthesis.py` | Template retrosynthesis, route ranking and recipe writer. |
| `app/streamlit_app.py` | Design page: English or slider input, molecule grid, metrics, table, per-molecule recipes, downloads. |
| `app/pages/1_Apply_a_formula.py` | Analyse a SMILES, formula or name: predictions, applicability, recipe. |
| `app/pages/2_Automation.py` | Readiness checklist, live model, run history, start a cycle, cron line. |

## Safety filters

Every molecule must pass all of these before it is scored or shown:

- only C, N, O, S, F, Cl, Br, I (plus H); no metals, phosphorus, boron, silicon, etc.
- a single connected fragment, no radicals
- molecular weight 150-600 and at most one Lipinski violation
- no PAINS or NIH structural alerts (assay-interfering, reactive and toxic motifs)
- no net charge (removable charges such as `[NH3+]` are neutralised first)
- a hazard screen drops anything matching controlled-substance scaffolds (4-anilidopiperidines,
  pethidine-type, amphetamine / cathinone / tryptamine types, barbiturates, benzodiazepines) or
  energetic / unstable groups (azides, peroxides, N-nitroso, polynitro)

The hazard screen also runs on every synthesis precursor: a route is discarded if one of its
building blocks matches. The English parser refuses requests that mention weapons, toxins,
explosives or controlled substances. These are coarse pattern screens, not a guarantee, and
they are not exhaustive.

Brenk alerts (unwanted functionality) do not reject a molecule but reduce its score.
The generator is trained only on drug-like ZINC molecules and these filters apply on top
of that. The system is scoped to drug-like small molecules.

## Honest limitations

- The solubility and logD models are trained on about 1,100 and 4,200 molecules. Treat
  their output as screening estimates, and use the `*_std` columns to spot molecules far
  from the training data.
- logP, QED and SA score are computed with RDKit's empirical models, not measured.
- Generated molecules are computational proposals. Nothing here predicts bioactivity,
  toxicity, ADMET or whether a compound can actually be made. Those need experimental
  work or purpose-built models.
- Recipes use generic textbook conditions for the reaction class. They are not optimised
  for the specific molecule, give no yield prediction, and do not check that precursors are
  commercially available. Confirm precedent in the literature, run a formal risk assessment,
  and work only in a properly equipped laboratory under a trained chemist. The targets are
  novel compounds of unknown toxicity.
- Routes are built only from the nine reaction classes, up to six steps. Protecting groups are not
  added automatically (a conflict is flagged instead), and nothing here predicts yield or checks that a
  starting material can actually be bought; molecules outside the classes get no recipe.

## Tests

```bash
python -m pytest tests -q
```
