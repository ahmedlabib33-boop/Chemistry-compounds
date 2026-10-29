"""Rule-based English request parser (no external model or API).

Turns a sentence such as "a water-soluble, easy-to-make, drug-like molecule with logP
around 3, give me 10" into design targets. Explicit numbers win over descriptive words.
Anything the generator cannot be conditioned on is reported back, not silently dropped,
and requests for weapons, toxic agents or controlled substances are refused.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .pipeline import DesignTargets

# Slider ranges used by the UI; parsed values are clamped to them.
LIMITS = {"logp": (-1.0, 6.0), "qed": (0.3, 0.95), "sas": (1.5, 6.0), "min_logs": (-8.0, 0.0)}

_NEGATED_TOXIC = re.compile(r"\b(?:non|not|un|anti|less|low|no)[\s-]*toxic(?:ity)?\b")
_BLOCKED = re.compile(
    r"\b(?:poison\w*|toxins?|toxic|nerve\s+agents?|chemical\s+weapons?|bio-?\s?weapons?|weapons?|"
    r"explosives?|bombs?|detonat\w*|lethal|deadly|kill\w*|fentanyl\w*|opioids?|opiates?|heroin|"
    r"cocaine|meth(?:amphetamine)?|amphetamines?|narcotics?|psychoactives?|hallucinogens?|"
    r"recreational\s+drugs?|street\s+drugs?|designer\s+drugs?|controlled\s+substances?|"
    r"date[\s-]rape|ghb|sarin|vx|novichok|tabun|soman|ricin|mustard\s+gas|overdose)\b")
REFUSAL = ("ChemGen designs drug-like screening molecules. It does not design weapons, toxic "
           "agents, explosives or controlled substances, so this request was not processed.")

# (pattern, value, label) tried in order; the first match wins.
_LOGP_WORDS = [
    (r"moderately\s+(?:lipophilic|hydrophobic)|balanced\s+lipophilicity", 2.5, "moderately lipophilic"),
    (r"very\s+(?:lipophilic|hydrophobic)|greasy|fat[\s-]soluble", 4.5, "very lipophilic"),
    (r"very\s+(?:hydrophilic|polar)", 0.0, "very hydrophilic"),
    (r"hydrophilic|polar|water[\s-]loving", 0.8, "hydrophilic"),
    (r"lipophilic|hydrophobic|membrane[\s-]permeable|cell[\s-]permeable", 3.5, "lipophilic"),
]
_QED_WORDS = [
    (r"(?:highly|very|extremely)\s+drug[\s-]?like", 0.9, "highly drug-like"),
    (r"drug[\s-]?like|orally|oral\b", 0.8, "drug-like"),
]
_SAS_WORDS = [
    (r"very\s+(?:easy|simple)|trivial", 2.0, "very easy to synthesise"),
    (r"(?:not|isn'?t|no)\s+(?:too\s+|overly\s+)?complex|easy|simple|cheap|straightforward|"
     r"synthesi[sz]able|accessible", 2.5, "easy to synthesise"),
    (r"complex|complicated|elaborate", 4.5, "complexity allowed"),
]
_LOGS_WORDS = [
    (r"(?:highly|very)\s+(?:water[\s-])?soluble", -2.0, "highly soluble"),
    (r"(?:poorly|insoluble)|not\s+soluble", -6.0, "poorly soluble allowed"),
    (r"soluble|solubility|aqueous", -3.0, "water soluble"),
]

# Things people ask for that the generator cannot be conditioned on.
_UNSUPPORTED = {
    r"molecular\s+weight|\bmw\b|heavy\s+atoms?|\bsmall\b|\blarge\b|\bbig\b":
        "size / molecular weight (steer indirectly with the QED and SA targets)",
    r"\bring\b|\brings\b|aromatic|chiral|stereo": "ring or stereochemistry counts",
    r"fluor|chlor|brom|iod|halogen|\bnitrogen\b|\bsulfur\b|\boxygen\b": "specific atoms or elements",
    r"kinase|receptor|inhibitor|binding|antibiotic|antiviral|antibacterial|anticancer|potency|"
    r"activity|selective|agonist|antagonist": "biological target or activity",
    r"toxicity|herg|cyp\d*|admet|half[\s-]life|stability|bioavailab": "toxicity / ADMET endpoints",
}


@dataclass
class ParsedRequest:
    targets: DesignTargets = field(default_factory=DesignTargets)
    top_k: int | None = None
    interpretation: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    refusal: str | None = None


def _clamp(name: str, value: float) -> tuple[float, bool]:
    lo, hi = LIMITS[name]
    clamped = min(max(value, lo), hi)
    return clamped, clamped != value


def _number(text: str, labels: str, signed: bool = False) -> float | None:
    sign = r"-?" if signed else r""
    m = re.search(rf"(?:{labels})[^\d\-.]{{0,18}}?({sign}\d*\.?\d+)", text)
    return float(m.group(1)) if m else None


def _word(text: str, table):
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

    if _BLOCKED.search(_NEGATED_TOXIC.sub(" ", lowered)):
        req.refusal = REFUSAL
        return req

    chosen = {}

    def set_target(key: str, label: str, value: float, source: str) -> None:
        value, clamped = _clamp(key, value)
        chosen[key] = value
        note = f"{label} = {value:g} ({source})"
        if clamped:
            note += f", limited to the supported range {LIMITS[key][0]:g} to {LIMITS[key][1]:g}"
        req.interpretation.append(note)

    numeric = {
        "logp": _number(lowered, r"log\s*p|lipophilicity", signed=True),
        "qed": _number(lowered, r"\bqed"),
        "sas": _number(lowered, r"\bsas\b|sa\s*score|synthetic\s+accessibility(?:\s+score)?"),
        "min_logs": _number(lowered, r"log\s*s(?![a-z])", signed=True),
    }
    if numeric["qed"] is not None and numeric["qed"] > 1:
        numeric["qed"] /= 100  # "qed 80" read as a percentage
    words = {
        "logp": _word(lowered, _LOGP_WORDS), "qed": _word(lowered, _QED_WORDS),
        "sas": _word(lowered, _SAS_WORDS), "min_logs": _word(lowered, _LOGS_WORDS),
    }
    labels = {"logp": "target logP", "qed": "minimum QED", "sas": "maximum SA score",
              "min_logs": "minimum logS"}
    for key in numeric:
        if numeric[key] is not None:
            set_target(key, labels[key], numeric[key], "stated value")
        elif words[key]:
            set_target(key, labels[key], words[key][0], f"from \"{words[key][1]}\"")

    m = re.search(r"\b(\d{1,3})\s+(?:molecules|compounds|candidates|structures|hits|options)\b", lowered) \
        or re.search(r"\btop\s+(\d{1,3})\b", lowered)
    if m:
        req.top_k = min(max(int(m.group(1)), 1), 50)
        req.interpretation.append(f"molecules to show = {req.top_k}")

    defaults = DesignTargets()
    req.targets = DesignTargets(
        logp=chosen.get("logp", defaults.logp), qed=chosen.get("qed", defaults.qed),
        sas=chosen.get("sas", defaults.sas), min_logs=chosen.get("min_logs", defaults.min_logs))
    for key, label in labels.items():
        if key not in chosen:
            req.interpretation.append(f"{label} = {getattr(req.targets, key):g} (default)")

    req.unsupported = [msg for pattern, msg in _UNSUPPORTED.items() if re.search(pattern, lowered)]
    return req
