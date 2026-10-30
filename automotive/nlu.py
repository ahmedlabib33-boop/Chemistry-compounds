"""Rule-based English request parser for the automotive domain (no external model or API; the
same discipline as chemgen.nlu). Maps a sentence like "a nonpolar base-oil-like hydrocarbon,
easy to make, show 10" onto AutomotiveTargets and a role filter. Numbers win over words. The
request-level refusal screen is shared, unchanged, with the drug app: this stays off-limits for
weapons, toxins, explosives and controlled substances regardless of domain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from chemgen.guard import REFUSAL, is_blocked

from .pipeline import AutomotiveTargets
from .roles import ROLES

LIMITS = {"logp": (-1.0, 10.0), "tpsa": (0.0, 120.0), "sas": (1.5, 6.0)}
ROLE_WORDS = {
    "corrosion_inhibitor": r"corrosion[\s-]?inhibitor|anti[\s-]?corrosion|corrosion[\s-]?protect\w*",
    "antiwear_ep": r"anti[\s-]?wear|antiwear|extreme[\s-]?pressure|\bep\b\s+additive",
    "antioxidant": r"antioxidant|anti[\s-]?oxidant|oxidation\s+stabili[sz]er",
    "friction_modifier": r"friction\s+modifier",
    "base_oil_like": r"base[\s-]?oil|lubricant\s+base|base\s+stock",
    "fuel_range_hydrocarbon": r"fuel[\s-]?(?:range|like)?|gasoline|petrol|diesel",
}
_LOGP_WORDS = [
    (r"very\s+(?:nonpolar|oil[\s-]?soluble)|highly\s+lipophilic", 6.0, "very nonpolar / oil-soluble"),
    (r"nonpolar|oil[\s-]?soluble|lipophilic|hydrocarbon[\s-]?like", 4.5, "nonpolar / oil-soluble"),
    (r"polar|water[\s-]?compatible|hydrophilic", 1.0, "polar"),
]
_TPSA_WORDS = [
    (r"very\s+nonpolar|purely\s+hydrocarbon", 5.0, "very low polarity"),
    (r"nonpolar|hydrocarbon[\s-]?like|base[\s-]?oil", 10.0, "low polarity"),
    (r"polar|functional(?:ised|ized)?\s+additive", 50.0, "more polar"),
]
_SAS_WORDS = [
    (r"very\s+(?:easy|simple)|trivial", 2.0, "very easy to synthesise"),
    (r"easy|simple|cheap|straightforward|accessible", 2.5, "easy to synthesise"),
    (r"complex|complicated|elaborate", 4.5, "complexity allowed"),
]
_UNSUPPORTED = {
    r"viscosity|viscous|kinematic\s+viscosity": "viscosity (no open dataset was found; not modelled)",
    r"flash\s*point": "flash point as a number (not modelled; boiling point and vapor pressure, which "
                      "it roughly tracks, are reported instead)",
    r"\bzddp\b|zinc\s+dial": "metal-containing additives (zinc, etc.) -- the generator is organic-only",
    r"pour\s+point|cold[\s-]?flow": "pour point directly (melting point, a related proxy, is reported)",
}


@dataclass
class ParsedRequest:
    targets: AutomotiveTargets = field(default_factory=AutomotiveTargets)
    top_k: int | None = None
    interpretation: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    refusal: str | None = None


def _clamp(name, value):
    lo, hi = LIMITS[name]
    c = min(max(value, lo), hi)
    return c, c != value


def _number(text, labels, signed=False):
    sign = r"-?" if signed else r""
    m = re.search(rf"(?:{labels})[^\d\-.]{{0,18}}?({sign}\d*\.?\d+)", text)
    return float(m.group(1)) if m else None


def _word(text, table):
    for pattern, value, label in table:
        if re.search(pattern, text):
            return value, label
    return None


def parse_request(text: str) -> ParsedRequest:
    req = ParsedRequest()
    lowered = " ".join((text or "").lower().split())
    if not lowered:
        req.interpretation.append("Empty description: using the default targets.")
        return req
    if is_blocked(lowered):
        req.refusal = REFUSAL
        return req

    chosen = {}

    def set_target(key, label, value, source):
        value, clamped = _clamp(key, value)
        chosen[key] = value
        note = f"{label} = {value:g} ({source})"
        if clamped:
            note += f", limited to {LIMITS[key][0]:g}-{LIMITS[key][1]:g}"
        req.interpretation.append(note)

    numeric = {"logp": _number(lowered, r"log\s*p|lipophilicity", signed=True),
               "tpsa": _number(lowered, r"tpsa|polar\s+surface\s+area"),
               "sas": _number(lowered, r"\bsas\b|sa\s*score|synthetic\s+accessibility(?:\s+score)?")}
    words = {"logp": _word(lowered, _LOGP_WORDS), "tpsa": _word(lowered, _TPSA_WORDS),
             "sas": _word(lowered, _SAS_WORDS)}
    labels = {"logp": "target logP", "tpsa": "target TPSA", "sas": "maximum SA score"}
    for key in numeric:
        if numeric[key] is not None:
            set_target(key, labels[key], numeric[key], "stated value")
        elif words[key]:
            set_target(key, labels[key], words[key][0], f'from "{words[key][1]}"')

    role = None
    for key, pattern in ROLE_WORDS.items():
        if re.search(pattern, lowered):
            role = key
            break
    if role:
        req.interpretation.append(f"role filter = {next(r.label for r in ROLES if r.key == role)}")

    m = (re.search(r"\b(\d{1,3})\s+(?:molecules|compounds|candidates|structures|hits|options)\b", lowered)
         or re.search(r"\btop\s+(\d{1,3})\b", lowered))
    if m:
        req.top_k = min(max(int(m.group(1)), 1), 50)
        req.interpretation.append(f"molecules to show = {req.top_k}")

    defaults = AutomotiveTargets()
    req.targets = AutomotiveTargets(logp=chosen.get("logp", defaults.logp), tpsa=chosen.get("tpsa", defaults.tpsa),
                                    sas=chosen.get("sas", defaults.sas), role=role)
    for key, label in labels.items():
        if key not in chosen:
            req.interpretation.append(f"{label} = {getattr(req.targets, key):g} (default)")

    req.unsupported = [msg for pattern, msg in _UNSUPPORTED.items() if re.search(pattern, lowered)]
    return req
