"""Request-level refusal screen shared by the English parser and the web harvesters."""
from __future__ import annotations

import re

_NEGATED_TOXIC = re.compile(r"\b(?:non|not|un|anti|less|low|no)[\s-]*toxic(?:ity)?\b")
_BLOCKED = re.compile(
    r"\b(?:poison\w*|toxins?|toxic|nerve\s+agents?|chemical\s+weapons?|bio-?\s?weapons?|weapons?|"
    r"explosives?|bombs?|detonat\w*|lethal|deadly|kill\w*|fentanyl\w*|opioids?|opiates?|heroin|"
    r"cocaine|meth(?:amphetamine)?|amphetamines?|narcotics?|psychoactives?|hallucinogens?|"
    r"recreational\s+drugs?|street\s+drugs?|designer\s+drugs?|controlled\s+substances?|"
    r"date[\s-]rape|ghb|sarin|vx|novichok|tabun|soman|ricin|mustard\s+gas|overdose)\b")
REFUSAL = ("ChemGen designs drug-like screening molecules. It does not design weapons, toxic "
           "agents, explosives or controlled substances, so this request was not processed.")


def is_blocked(text: str) -> bool:
    """True if the text asks for weapons, toxic agents, explosives or controlled substances."""
    return bool(_BLOCKED.search(_NEGATED_TOXIC.sub(" ", " ".join(text.lower().split()))))
