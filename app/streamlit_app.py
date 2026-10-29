"""Web UI: describe a molecule in English (or set sliders), generate, then inspect ranked
results with a step-by-step synthesis recipe for each molecule."""
import sys
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_pipeline, model_stamp  # noqa: E402
from chemgen.nlu import parse_request  # noqa: E402
from chemgen.pipeline import DesignTargets  # noqa: E402

st.set_page_config(page_title="ChemGen", layout="wide")

EXAMPLE = ("An easy to make, water soluble, drug-like molecule with logP around 2.5. "
           "Show me 10 molecules.")


st.title("ChemGen: drug-like molecule design")
st.caption("Describe what you need. A property-conditioned SMILES generator proposes molecules; "
           "random-forest models predict solubility and lipophilicity; safety filters and a "
           "desirability score rank them; each result comes with a one-step synthesis recipe.")

try:
    pipeline = load_pipeline(model_stamp())
except FileNotFoundError:
    st.error("Trained models not found. Run `python -m chemgen download`, "
             "`python -m chemgen train-predictors` and `python -m chemgen train-generator` first.")
    st.stop()

with st.sidebar:
    mode = st.radio("Input", ["Describe in English", "Sliders"], horizontal=True)
    if mode == "Describe in English":
        description = st.text_area("What do you need?", value=EXAMPLE, height=130,
                                   help="You can state values (logP 3, QED 0.85, SAS 2.5, logS -3) "
                                        "or use words (lipophilic, drug-like, easy to make, water soluble).")
        st.caption("Understood: logP, drug-likeness (QED), synthetic accessibility, solubility and "
                   "how many molecules to show. Size, atoms, rings and biological targets are not modelled.")
    else:
        logp = st.slider("logP (lipophilicity)", -1.0, 6.0, 2.5, 0.1)
        qed = st.slider("Minimum QED (drug-likeness)", 0.3, 0.95, 0.8, 0.01)
        sas = st.slider("Maximum SA score (1 easy, 10 hard)", 1.5, 6.0, 3.0, 0.1)
        min_logs = st.slider("Minimum solubility, logS (mol/L)", -8.0, 0.0, -4.0, 0.1)
    st.header("Search")
    n_samples = st.select_slider("Molecules to sample", [500, 1000, 2000, 5000, 10000], value=2000)
    top_k = st.slider("Molecules to show", 5, 50, 20)
    temperature = st.slider("Sampling temperature", 0.5, 1.5, 1.0, 0.05,
                            help="Lower is safer and more conventional; higher is more novel but less valid.")
    seed = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)
    run = st.button("Generate", type="primary", width="stretch")

if run:
    request = None
    if mode == "Describe in English":
        request = parse_request(description)
        if request.refusal:
            st.session_state.pop("result", None)
            st.session_state["request"] = request
        else:
            targets, shown = request.targets, request.top_k or top_k
    else:
        targets, shown = DesignTargets(logp, qed, sas, min_logs), top_k
    if request is None or not request.refusal:
        with st.spinner("Sampling, filtering, scoring and planning syntheses..."):
            st.session_state["result"] = pipeline.design(
                targets, n_samples=n_samples, top_k=shown, temperature=temperature,
                seed=int(seed) or None)
        st.session_state["request"] = request

request = st.session_state.get("request")
if request is not None:
    if request.refusal:
        st.error(request.refusal)
        st.stop()
    with st.expander("How I read your request", expanded=True):
        for line in request.interpretation:
            st.markdown(f"- {line}")
        for line in request.unsupported:
            st.warning(f"Not modelled, so ignored: {line}")

result = st.session_state.get("result")
if result is None:
    st.info("Describe the molecule you want (or switch to sliders) and press Generate.")
    st.stop()

cols = st.columns(5)
for col, (label, key) in zip(cols, [("Sampled", "sampled"), ("Valid", "valid"), ("Unique", "unique"),
                                    ("Novel", "novel"), ("Passed safety", "passed_safety")]):
    col.metric(label, result.stats[key])

if result.table.empty:
    st.warning("No molecules passed the filters. Sample more molecules or lower the temperature.")
    st.stop()

table = result.table
mols = [Chem.MolFromSmiles(s) for s in table["SMILES"]]
legends = [f"#{i + 1}  score {r.score:.2f}  logP {r.logP:.1f}  QED {r.QED:.2f}"
           for i, r in enumerate(table.itertuples())]
st.image(Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 240), legends=legends),
         width="stretch")

st.dataframe(table.round(3), width="stretch", hide_index=True)
st.download_button("Download table (CSV)", table.to_csv(index=False), "designed_molecules.csv", "text/csv")

st.header("Synthesis recipes")
st.caption("One-step routes from six common reaction classes (amide coupling, sulfonamide formation, "
           "reductive amination, SNAr, Suzuki coupling, aryl-ether alkylation). Conditions are generic "
           "textbook procedures, not optimised for these molecules and not experimentally validated.")
st.download_button("Download all recipes (Markdown)", result.recipes_markdown(), "recipes.md", "text/markdown")
for i, smiles in enumerate(table["SMILES"], 1):
    recipe = result.recipes.get(smiles)
    title = f"#{i}  {smiles}  ·  {recipe.reaction_name if recipe else 'no one-step route'}"
    with st.expander(title):
        if recipe is None:
            st.write("None of the supported reaction classes disconnects this molecule in one step.")
            continue
        precursors = [Chem.MolFromSmiles(recipe.electrophile_smiles),
                      Chem.MolFromSmiles(recipe.nucleophile_smiles), mols[i - 1]]
        st.image(Draw.MolsToGridImage(precursors, molsPerRow=3, subImgSize=(260, 200),
                                      legends=["precursor A", "precursor B", "target"]))
        st.markdown(recipe.to_markdown())

st.caption("Predicted solubility and logD come from models trained on small public datasets (about 1,100 "
           "and 4,200 molecules); treat them as screening estimates. Generated structures and recipes "
           "are computational proposals, not validated compounds or procedures.")
