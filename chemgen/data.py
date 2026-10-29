"""Open datasets used for training."""
from __future__ import annotations

import urllib.request
from pathlib import Path

DATASET_URLS = {
    "delaney.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "lipophilicity.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
    "zinc250k.csv": ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/"
                     "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"),
}


def download(data_dir: Path, force: bool = False, log=print) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DATASET_URLS.items():
        target = data_dir / filename
        if target.exists() and not force:
            log(f"{filename}: already present")
            continue
        log(f"{filename}: downloading")
        urllib.request.urlretrieve(url, target)
