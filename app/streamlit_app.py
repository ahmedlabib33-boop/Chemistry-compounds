"""Web UI: pick target properties, generate molecules, inspect and download the ranked results."""
import sys
from pathlib import Path

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chemgen import MODEL_DIR  # noqa: E402
from chemgen.pipeline import DesignPipeline, DesignTargets  # noqa: E402

st.set_page_config(page_title="ChemGen", layout="wide")


@st.cache_resource
def load_pipeline() -> DesignPipeline:
    return DesignPipeline(MODEL_DIR)


st.title("ChemGen: drug-like molecule design")
st.caption("A property-conditioned SMILES generator proposes molecules; random-forest models predict "
           "solubility and lipophilicity; safety filters and a desirability score rank the results.")

try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("Trained models not found. Run `python -m chemgen download`, "
             "`python -m chemgen train-predictors` and `python -m chemgen train-generator` first.")
    st.stop()

with st.sidebar:
    st.header("Target properties")
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
    run = st.button("Generate", type="primary", use_container_width=True)

if run:
    with st.spinner("Sampling, filtering and scoring..."):
        result = pipeline.design(DesignTargets(logp, qed, sas, min_logs), n_samples=n_samples,
                                 top_k=top_k, temperature=temperature, seed=int(seed) or None)
    st.session_state["result"] = result

result = st.session_state.get("result")
if result is None:
    st.info("Choose target properties in the sidebar and press Generate.")
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
         use_container_width=True)

st.dataframe(table.round(3), use_container_width=True, hide_index=True)
st.download_button("Download CSV", table.to_csv(index=False), "designed_molecules.csv", "text/csv")
st.caption("Predicted solubility and logD come from models trained on small public datasets (about 1,100 "
           "and 4,200 molecules). Treat them as screening estimates; the std columns show disagreement "
           "between trees. Generated structures are computational proposals, not validated compounds.")
