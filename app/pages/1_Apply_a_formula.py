"""Apply the trained models to a structure, molecular formula or compound name you supply."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from common import load_bundle, model_stamp  # noqa: E402
from chemgen import DATA_DIR  # noqa: E402
from chemgen.apply import analyze  # noqa: E402
from chemgen.harvest import HttpClient, HttpError, PubChemSource, Refused, candidates  # noqa: E402

st.set_page_config(page_title="Apply a formula", layout="wide")
st.title("Apply a formula")
st.caption("Give the app a structure, a molecular formula or a name. It screens the structure, "
           "predicts solubility and lipophilicity with an uncertainty and an applicability check, "
           "and writes out how to prepare it.")

try:
    bundle = load_bundle(model_stamp())
except FileNotFoundError:
    st.error("Models are not trained yet. Run `python -m chemgen prepare` (see the Automation page).")
    st.stop()

KINDS = {"Structure (SMILES)": ("smiles", "CC(=O)Nc1ccc(O)cc1"),
         "Molecular formula": ("formula", "C9H8O4"),
         "Compound name": ("name", "ibuprofen")}
choice = st.radio("I have a", list(KINDS), horizontal=True)
kind, example = KINDS[choice]
value = st.text_input("Value", value=example, key=f"value_{kind}")
cols = st.columns(3)
scale = cols[0].number_input("Final-step scale (mmol)", 0.1, 100.0, 1.0, 0.1)
max_steps = cols[1].slider("Maximum synthesis steps", 1, 5, 3)
limit = cols[2].slider("Most structures to analyse (formula / name)", 1, 10, 3, disabled=kind == "smiles")
check = st.checkbox("Check starting materials on PubChem (needs internet)", value=False)

if st.button("Analyse", type="primary"):
    smiles_list, error = [], None
    if kind == "smiles":
        smiles_list = [value]
    else:
        try:
            with st.spinner("Looking it up on PubChem..."):
                records = candidates(kind, value, limit=limit, client=HttpClient(cache_dir=DATA_DIR / "cache"))
            smiles_list = [r.smiles for r in records[:limit]]
            if not smiles_list:
                error = "PubChem returned nothing for that query."
        except Refused as err:
            error = str(err)
        except (ValueError, HttpError) as err:
            error = str(err)
    lookup = PubChemSource(HttpClient(cache_dir=DATA_DIR / "cache")).cid_for_smiles if check else None
    st.session_state["apply"] = {"error": error, "analyses": [
        analyze(s, bundle, scale_mmol=scale, max_steps=max_steps, known_lookup=lookup) for s in smiles_list]}

state = st.session_state.get("apply")
if state is None:
    st.info("Enter a value and press Analyse.")
    st.stop()
if state["error"]:
    st.warning(state["error"])
    st.stop()

for i, analysis in enumerate(state["analyses"], 1):
    label = {"ok": "", "refused": "  ·  refused", "unsupported": "  ·  not supported",
             "invalid": "  ·  invalid"}[analysis.status]
    with st.expander(f"#{i}  {analysis.smiles or analysis.input}{label}", expanded=len(state["analyses"]) == 1):
        if analysis.status == "ok":
            st.image(Draw.MolToImage(Chem.MolFromSmiles(analysis.smiles), size=(320, 240)))
        st.markdown(analysis.to_markdown())
st.download_button("Download report (Markdown)", "\n\n---\n\n".join(a.to_markdown() for a in state["analyses"]),
                   "analysis.md", "text/markdown")
