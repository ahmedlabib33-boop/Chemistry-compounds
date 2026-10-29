"""Shared Streamlit loaders. The cache key includes the live model's timestamp, so promoting a
new generator (``python -m chemgen automate``) is picked up without restarting the app."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from chemgen import DATA_DIR, MODEL_DIR  # noqa: E402
from chemgen.apply import ModelBundle  # noqa: E402
from chemgen.pipeline import DesignPipeline  # noqa: E402


def model_stamp() -> float:
    path = MODEL_DIR / "generator.pt"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource
def load_pipeline(stamp: float) -> DesignPipeline:
    return DesignPipeline(MODEL_DIR)


@st.cache_resource
def load_bundle(stamp: float) -> ModelBundle:
    return ModelBundle.load(MODEL_DIR, DATA_DIR)
