"""automotive: ML-driven design of automotive chemicals (fuel/lubricant additives, corrosion
inhibitors, base-oil-like hydrocarbons) -- a sibling app to chemgen's drug-like molecule design,
reusing its SMILES tokenizer, generator architecture, hazard screens, retrosynthesis planner and
harvesting agents, with its own data, property models and role classifier for this domain.
"""
from pathlib import Path

from chemgen import ROOT

DATA_DIR = ROOT / "data" / "automotive"
MODEL_DIR = ROOT / "models" / "automotive"
