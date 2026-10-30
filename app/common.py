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


def automotive_model_stamp() -> float:
    from automotive import MODEL_DIR as AUTO_MODEL_DIR
    path = AUTO_MODEL_DIR / "generator.pt"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource
def load_automotive_pipeline(stamp: float):
    from automotive import DATA_DIR as AUTO_DATA_DIR
    from automotive import MODEL_DIR as AUTO_MODEL_DIR
    from automotive.pipeline import AutomotivePipeline
    return AutomotivePipeline(AUTO_MODEL_DIR, AUTO_DATA_DIR)


@st.cache_resource
def load_automotive_bundle(stamp: float):
    from automotive import DATA_DIR as AUTO_DATA_DIR
    from automotive import MODEL_DIR as AUTO_MODEL_DIR
    from automotive.apply import ModelBundle as AutomotiveModelBundle
    return AutomotiveModelBundle.load(AUTO_MODEL_DIR, AUTO_DATA_DIR)
