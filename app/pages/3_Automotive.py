"""Automotive chemicals: design new candidates, or apply the models to a structure you supply.
Boiling point, vapor pressure, melting point and water solubility are real ML predictions,
trained on EPA OPERA measurements. Automotive "role" tags are rule-based functional-group
matches, not measured activity. Flash point and viscosity are not modelled (no open dataset)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

from common import (automotive_model_stamp, load_automotive_bundle,  # noqa: E402
                    load_automotive_pipeline)
from automotive.apply import analyze  # noqa: E402
from automotive.nlu import parse_request  # noqa: E402
from automotive.pipeline import AutomotiveTargets  # noqa: E402
from automotive.prepare import is_ready, run_checks  # noqa: E402
from automotive.roles import ROLES  # noqa: E402

st.set_page_config(page_title="Automotive chemistry", layout="wide")
st.title("Automotive chemistry")
st.caption("A property-conditioned SMILES generator, trained on EPA OPERA measurements instead of a "
           "drug-discovery set, proposes fuel/lubricant-style molecules. Boiling point, vapor pressure, "
           "melting point and water solubility are real ML predictions. Flash point and viscosity are "
           "not modelled: no open, downloadable dataset for either was found. Automotive-role tags "
           "(corrosion inhibitor, antiwear, antioxidant, ...) are rule-based functional-group matches, "
           "not measured or learned activity.")

checks = run_checks()
if not is_ready(checks):
    st.error("Not ready. Run `python -m automotive prepare` (add `--quick` for a faster first pass), "
             "then reload this page.")
    with st.expander("Readiness detail"):
        for c in checks:
            st.markdown(("✅ " if c.ok else "❌ ") + f"**{c.name}**: {c.detail}"
                       + (f"  \n&nbsp;&nbsp;&nbsp;&nbsp;fix: `{c.fix}`" if not c.ok and c.fix else ""))
    st.stop()

tab_design, tab_apply = st.tabs(["Design new chemicals", "Apply a formula"])

ROLE_OPTIONS = {"Any": None, **{r.label: r.key for r in ROLES}}

with tab_design:
    mode = st.radio("Input", ["Describe in English", "Sliders"], horizontal=True, key="auto_mode")
    if mode == "Describe in English":
        description = st.text_area("What do you need?", key="auto_text", height=100,
                                   value="A nonpolar, base-oil-like hydrocarbon, easy to make, show 10")
        st.caption("Understood: logP, TPSA (polarity), synthetic accessibility, an automotive role "
                   "(corrosion inhibitor, antiwear, antioxidant, friction modifier, base-oil-like, "
                   "fuel-range) and how many to show. Viscosity, flash point and metal-containing "
                   "additives (e.g. ZDDP) are not modelled.")
    else:
        logp = st.slider("Target logP", -1.0, 10.0, 4.0, 0.1, key="auto_logp")
        tpsa = st.slider("Target TPSA (polarity; low = base-oil/fuel-like)", 0.0, 120.0, 20.0, 1.0, key="auto_tpsa")
        sas = st.slider("Maximum SA score", 1.5, 6.0, 3.0, 0.1, key="auto_sas")
        role_label = st.selectbox("Automotive role filter", list(ROLE_OPTIONS), key="auto_role")
    n_samples = st.select_slider("Molecules to sample", [500, 1000, 2000, 5000], value=2000, key="auto_n")
    top_k = st.slider("Molecules to show", 5, 50, 20, key="auto_top")
    max_steps = st.slider("Maximum synthesis steps", 1, 5, 3, key="auto_steps")
    seed = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1, key="auto_seed")

    if st.button("Generate", type="primary", key="auto_generate"):
        request = None
        if mode == "Describe in English":
            request = parse_request(description)
            if request.refusal:
                st.session_state.pop("auto_result", None)
                st.session_state["auto_request"] = request
            else:
                targets, shown = request.targets, request.top_k or top_k
        else:
            targets = AutomotiveTargets(logp, tpsa, sas, ROLE_OPTIONS[role_label])
            shown = top_k
        if request is None or not request.refusal:
            with st.spinner("Sampling, filtering, scoring and planning syntheses..."):
                pipeline = load_automotive_pipeline(automotive_model_stamp())
                st.session_state["auto_result"] = pipeline.design(
                    targets, n_samples=n_samples, top_k=shown, seed=int(seed) or None, max_steps=max_steps)
            st.session_state["auto_request"] = request

    request = st.session_state.get("auto_request")
    if request is not None:
        if request.refusal:
            st.error(request.refusal)
        else:
            with st.expander("How I read your request", expanded=True):
                for line in request.interpretation:
                    st.markdown(f"- {line}")
                for line in request.unsupported:
                    st.warning(f"Not modelled, so ignored: {line}")

    result = st.session_state.get("auto_result")
    if result is not None and not (request and request.refusal):
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("Sampled", "sampled"), ("Valid", "valid"), ("Unique", "unique"),
                                            ("Novel", "novel"), ("Passed safety", "passed_safety")]):
            col.metric(label, result.stats[key])
        if result.table.empty:
            st.warning("No molecules passed the filters. Sample more, or widen the targets.")
        else:
            table = result.table
            mols = [Chem.MolFromSmiles(s) for s in table["SMILES"]]
            legends = [f"#{i + 1}  score {r.score:.2f}  logP {r.logP:.1f}  TPSA {r.TPSA:.0f}\n{r.role_tags}"
                      for i, r in enumerate(table.itertuples())]
            st.image(Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 260), legends=legends),
                    width="stretch")
            st.dataframe(table.drop(columns=["roles"]).round(3), width="stretch", hide_index=True)
            st.download_button("Download table (CSV)", table.drop(columns=["roles"]).to_csv(index=False),
                              "automotive_molecules.csv", "text/csv")
            st.download_button("Download all recipes (Markdown)", result.recipes_markdown(),
                              "automotive_recipes.md", "text/markdown")
            for i, smiles in enumerate(table["SMILES"], 1):
                recipe = result.recipes.get(smiles)
                with st.expander(f"#{i}  {smiles}  ·  {recipe.summary if recipe else 'no route found'}"):
                    if recipe is None:
                        st.write("None of the supported reaction classes reaches this molecule within "
                                "the step limit.")
                    else:
                        st.markdown(recipe.to_markdown())

with tab_apply:
    value = st.text_input("SMILES", value="Cc1ccccc1OP(=O)(Oc1ccccc1C)Oc1ccccc1C", key="auto_apply_smiles")
    cols = st.columns(2)
    scale = cols[0].number_input("Final-step scale (mmol)", 0.1, 100.0, 1.0, 0.1, key="auto_apply_scale")
    steps = cols[1].slider("Maximum synthesis steps", 1, 5, 3, key="auto_apply_steps")
    if st.button("Analyse", type="primary", key="auto_apply_go"):
        bundle = load_automotive_bundle(automotive_model_stamp())
        st.session_state["auto_apply_result"] = analyze(value, bundle, scale_mmol=scale, max_steps=steps)
    analysis = st.session_state.get("auto_apply_result")
    if analysis is not None:
        if analysis.status == "ok":
            st.image(Draw.MolToImage(Chem.MolFromSmiles(analysis.smiles), size=(320, 240)))
        st.markdown(analysis.to_markdown())
