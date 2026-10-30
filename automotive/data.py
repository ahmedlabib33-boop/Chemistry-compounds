"""Training data: the EPA's OPERA QSAR suite (public-domain measurements, MIT-licensed repo),
mirrored at github.com/kmansouri/OPERA. We use four of its curated property sets: boiling point
(BP), vapor pressure (VP), melting point (MP) and water solubility (WS) -- all measured physical
properties of real, mostly non-drug-like industrial and environmental chemicals, which is a much
better match for automotive chemistry's chemical space than a drug-discovery set like ZINC.

No open, programmatically-downloadable dataset of flash point, viscosity, or corrosion-inhibition
/ antiwear activity was found. Those are not modelled with ML here -- see automotive/roles.py for
the honest, rule-based substitute for additive "role" and automotive/predictor.py's docstring for
why flash point and viscosity are not predicted.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

OPERA_ZIP_URL = "https://raw.githubusercontent.com/kmansouri/OPERA/master/OPERA_Data.zip"
OPERA_LICENSE_URL = "https://raw.githubusercontent.com/kmansouri/OPERA/master/LICENSE"

# member name inside OPERA_Data.zip -> file this module writes into DATA_DIR
MEMBERS = {
    "BP_2.9_update.csv": "bp.csv",
    "VP_2.9_update.csv": "vp.csv",
    "MP_2.9_update.csv": "mp.csv",
    "WS_2.9_update.csv": "ws.csv",
}


def download(data_dir: Path, force: bool = False, log=print) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / out).exists() for out in MEMBERS.values()) and not force:
        log("OPERA property data: already present")
        return
    log("OPERA_Data.zip: downloading (public-domain measurements, MIT-licensed repo)")
    with urllib.request.urlopen(OPERA_ZIP_URL, timeout=180) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for member, out_name in MEMBERS.items():
        with archive.open(member) as src, (data_dir / out_name).open("wb") as dst:
            dst.write(src.read())
        log(f"{out_name}: extracted from {member}")
    try:
        urllib.request.urlretrieve(OPERA_LICENSE_URL, data_dir / "OPERA_LICENSE")
    except OSError:
        pass  # not essential; the licence text is also quoted in this repo's README
