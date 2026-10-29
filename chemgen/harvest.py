"""Retrieval agents that pull molecules from free public chemistry databases.

Four kinds of agent cooperate, all in plain Python with no paid services or API keys:

* ``SourceAgent`` streams raw records from one database (PubChem or ChEMBL) through its
  official REST API, politely: rate-limited, retried with back-off, cached on disk.
* ``Curator`` standardises every record (largest fragment, neutralised, canonical SMILES) and
  applies the same safety screens as the design pipeline. Anything that fails never reaches
  the model's training data.
* ``Store`` deduplicates and appends accepted molecules with their provenance.
* ``harvest`` is the coordinator: it runs one SourceAgent per database in parallel.

The output CSV has the same columns as the ZINC file, so it can be mixed into generator
training (``train-generator --extra``).

Scope: this reads databases that publish an API, not arbitrary web pages. Free-text pages
are unreliable (typos, wrong formulas, unclear licences) and cannot be verified. Add a new
database by writing a class with ``name`` and ``stream()`` like the two below.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from rdkit import Chem

from . import chem, safety
from .guard import REFUSAL, is_blocked
from .tokenizer import MAX_TOKENS, tokenize

USER_AGENT = "chemgen-research/0.1 (open-source cheminformatics; github.com/ahmedlabib33-boop/Chemistry-compounds)"
FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d{0,4}){1,12}$")


class HttpError(RuntimeError):
    pass


class Refused(ValueError):
    """A query was declined by the request-level screen."""


class HttpClient:
    """Minimal JSON client: one request at a time per client, retries, on-disk cache."""

    def __init__(self, min_interval: float = 0.25, cache_dir: Path | None = None, retries: int = 3,
                 timeout: float = 60.0, opener: urllib.request.OpenerDirector | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval, self.retries, self.timeout = min_interval, retries, timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.opener = opener or urllib.request.build_opener()
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last = 0.0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()

    def get_json(self, url: str, data: dict | None = None):
        """GET (or form-encoded POST when ``data`` is given). Returns None on HTTP 404."""
        body = urllib.parse.urlencode(data).encode() if data else None
        cache_file = None
        if self.cache_dir:
            key = hashlib.sha1(url.encode() + (body or b"")).hexdigest()
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                return json.loads(cache_file.read_text())
        for attempt in range(self.retries + 1):
            self._throttle()
            request = urllib.request.Request(
                url, data=body, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    payload = json.load(response)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    return None
                if err.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    retry_after = err.headers.get("Retry-After") if err.headers else None
                    self._sleep(float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt)
                    continue
                raise HttpError(f"HTTP {err.code} for {url}") from err
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                if attempt < self.retries:
                    self._sleep(2 ** attempt)
                    continue
                raise HttpError(f"{type(err).__name__} for {url}: {err}") from err
            if cache_file is not None:
                cache_file.write_text(json.dumps(payload))
            return payload
        raise HttpError(f"gave up on {url}")  # pragma: no cover


@dataclass(frozen=True)
class RawRecord:
    smiles: str
    source: str
    source_id: str
    formula: str | None = None


class PubChemSource:
    """PubChem PUG REST (public domain; asks for at most 5 requests per second)."""

    name = "pubchem"
    BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    PROPS = "SMILES,MolecularFormula"

    def __init__(self, client: HttpClient, max_cid: int = 20_000_000, batch: int = 200,
                 rng: random.Random | None = None):
        self.client, self.max_cid, self.batch = client, max_cid, batch
        self.rng = rng or random.Random()

    def _records(self, cids: list[int]) -> list[RawRecord]:
        payload = self.client.get_json(f"{self.BASE}/compound/cid/property/{self.PROPS}/JSON",
                                       data={"cid": ",".join(map(str, cids))})
        rows = ((payload or {}).get("PropertyTable") or {}).get("Properties", [])
        return [RawRecord(r["SMILES"], self.name, str(r["CID"]), r.get("MolecularFormula"))
                for r in rows if r.get("SMILES")]

    def _cids(self, url: str, data: dict | None = None) -> list[int]:
        payload = self.client.get_json(url, data=data)
        for _ in range(6):  # large searches are answered asynchronously with a ListKey
            waiting = (payload or {}).get("Waiting")
            if not waiting:
                break
            self.client._sleep(2.0)
            payload = self.client.get_json(f"{self.BASE}/compound/listkey/{waiting['ListKey']}/cids/JSON")
        return list(((payload or {}).get("IdentifierList") or {}).get("CID", []))

    def stream(self) -> Iterator[RawRecord]:
        """Random CIDs: a broad, unbiased sample of what PubChem holds."""
        while True:
            yield from self._records(self.rng.sample(range(1, self.max_cid), self.batch))

    def by_formula(self, formula: str, limit: int) -> Iterator[RawRecord]:
        cids = self._cids(f"{self.BASE}/compound/fastformula/{urllib.parse.quote(formula)}/cids/JSON"
                          f"?MaxRecords={limit}")
        for i in range(0, len(cids), self.batch):
            yield from self._records(cids[i:i + self.batch])

    def similar_to(self, smiles: str, threshold: int, limit: int) -> Iterator[RawRecord]:
        cids = self._cids(f"{self.BASE}/compound/fastsimilarity_2d/smiles/cids/JSON"
                          f"?Threshold={threshold}&MaxRecords={limit}", data={"smiles": smiles})
        for i in range(0, len(cids), self.batch):
            yield from self._records(cids[i:i + self.batch])

    def cid_for_smiles(self, smiles: str) -> int | None:
        """PubChem CID of this exact structure, or None. A known compound is not necessarily purchasable."""
        payload = self.client.get_json(f"{self.BASE}/compound/smiles/cids/JSON", data={"smiles": smiles})
        cids = ((payload or {}).get("IdentifierList") or {}).get("CID", [])
        return cids[0] if cids and cids[0] else None

    def by_name(self, name: str) -> Iterator[RawRecord]:
        payload = self.client.get_json(
            f"{self.BASE}/compound/name/{urllib.parse.quote(name)}/property/{self.PROPS}/JSON")
        for r in ((payload or {}).get("PropertyTable") or {}).get("Properties", []):
            if r.get("SMILES"):
                yield RawRecord(r["SMILES"], self.name, str(r["CID"]), r.get("MolecularFormula"))


class ChEMBLSource:
    """ChEMBL web services (CC BY-SA 3.0: cite ChEMBL if you share data derived from it)."""

    name = "chembl"
    BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

    def __init__(self, client: HttpClient, page_size: int = 500, rng: random.Random | None = None):
        self.client, self.page_size = client, page_size
        self.rng = rng or random.Random()

    def _query(self, offset: int, limit: int) -> str:
        params = {"limit": limit, "offset": offset,
                  "only": "molecule_chembl_id,molecule_structures",
                  "molecule_properties__full_mwt__gte": 150, "molecule_properties__full_mwt__lte": 600}
        return f"{self.BASE}?{urllib.parse.urlencode(params)}"

    def stream(self) -> Iterator[RawRecord]:
        """Pages of drug-sized molecules, starting at a random offset."""
        head = self.client.get_json(self._query(0, 1)) or {}
        total = (head.get("page_meta") or {}).get("total_count", 0)
        offset = self.rng.randrange(0, max(1, total - self.page_size)) if total else 0
        while True:
            page = self.client.get_json(self._query(offset, self.page_size)) or {}
            molecules = page.get("molecules", [])
            for m in molecules:
                smiles = (m.get("molecule_structures") or {}).get("canonical_smiles")
                if smiles:
                    yield RawRecord(smiles, self.name, m["molecule_chembl_id"])
            if not molecules or not (page.get("page_meta") or {}).get("next"):
                return
            offset += self.page_size


def _reason_key(reason: str) -> str:
    return re.split(r"[:\d(]", reason)[0].strip()


class Curator:
    """Standardise a raw record and apply the design pipeline's safety screens."""

    def curate(self, smiles: str) -> tuple[dict | None, str]:
        mol = chem.to_mol(smiles)
        if mol is None:
            return None, "unparseable"
        mol = chem.standardize(mol)
        canonical = Chem.MolToSmiles(mol)
        try:
            if len(tokenize(canonical)) > MAX_TOKENS:
                return None, "too long"
        except ValueError:
            return None, "untokenizable"
        verdict = safety.assess(mol)
        if not verdict.passed:
            return None, _reason_key(verdict.reasons[0])
        props = chem.basic_properties(mol)
        return {"smiles": canonical, "logP": round(props["logP"], 4), "qed": round(props["QED"], 4),
                "SAS": round(props["SAS"], 4)}, "ok"


class Store:
    """Thread-safe, append-only CSV of accepted molecules with provenance."""

    FIELDS = ["smiles", "logP", "qed", "SAS", "source", "source_id", "fetched_at"]

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        if self.path.exists():
            import csv
            with self.path.open(newline="") as f:
                self._seen = {row["smiles"] for row in csv.DictReader(f)}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(",".join(self.FIELDS) + "\n")

    def __len__(self) -> int:
        return len(self._seen)

    def add(self, row: dict) -> bool:
        import csv
        with self._lock:
            if row["smiles"] in self._seen:
                return False
            with self.path.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
            self._seen.add(row["smiles"])
            return True


@dataclass
class AgentReport:
    name: str
    fetched: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejected: Counter = field(default_factory=Counter)
    error: str | None = None
    seconds: float = 0.0

    def line(self) -> str:
        top = ", ".join(f"{k} {v}" for k, v in self.rejected.most_common(4)) or "none"
        status = f"  STOPPED: {self.error}" if self.error else ""
        return (f"{self.name}: fetched {self.fetched}, accepted {self.accepted}, duplicates "
                f"{self.duplicates}, rejected {sum(self.rejected.values())} ({top}) "
                f"in {self.seconds:.0f}s{status}")


class SourceAgent:
    def __init__(self, source, curator: Curator, store: Store, budget: int,
                 max_fetched: int | None = None):
        self.source, self.curator, self.store = source, curator, store
        self.budget, self.max_fetched = budget, max_fetched or budget * 25

    def run(self) -> AgentReport:
        report, started = AgentReport(self.source.name), time.monotonic()
        try:
            for record in self.source.stream():
                report.fetched += 1
                row, reason = self.curator.curate(record.smiles)
                if row is None:
                    report.rejected[reason] += 1
                else:
                    row.update(source=record.source, source_id=record.source_id,
                               fetched_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
                    if self.store.add(row):
                        report.accepted += 1
                    else:
                        report.duplicates += 1
                if report.accepted >= self.budget or report.fetched >= self.max_fetched:
                    break
        except HttpError as err:  # network trouble ends this agent, not the others
            report.error = str(err)
        report.seconds = time.monotonic() - started
        return report


SOURCES = {"pubchem": (PubChemSource, 0.25), "chembl": (ChEMBLSource, 0.25)}


def harvest(sources: list[str], budget: int, out_path: Path, cache_dir: Path | None = None,
            seed: int = 0, log=print) -> list[AgentReport]:
    """Run one agent per source in parallel; each stops after ``budget`` new molecules."""
    store, curator = Store(out_path), Curator()
    agents = []
    for i, name in enumerate(sources):
        cls, interval = SOURCES[name]
        client = HttpClient(min_interval=interval, cache_dir=cache_dir)
        agents.append(SourceAgent(cls(client, rng=random.Random(seed + i)), curator, store, budget))
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        reports = list(pool.map(lambda a: a.run(), agents))
    for report in reports:
        log(report.line())
    log(f"{out_path}: {len(store)} molecules in total")
    return reports


@dataclass
class LookupResult:
    accepted: list[dict]
    rejected: Counter


def candidates(kind: str, value: str, limit: int = 25, threshold: int = 90,
               client: HttpClient | None = None) -> list[RawRecord]:
    """Raw PubChem records for a formula, name or similar-structure query (screened queries only)."""
    value = value.strip()
    if kind == "formula":
        if not FORMULA.match(value):
            raise ValueError(f"not a molecular formula: {value!r} (use e.g. C9H8O4)")
    elif is_blocked(value):
        raise Refused(REFUSAL)
    if kind == "similar":
        query = chem.to_mol(value)
        if query is None:
            raise ValueError(f"cannot parse SMILES: {value!r}")
        if safety.hazard_flags(query):
            raise Refused(REFUSAL)
    source = PubChemSource(client or HttpClient(cache_dir=None))
    return list({"formula": lambda: source.by_formula(value, limit),
                 "name": lambda: source.by_name(value),
                 "similar": lambda: source.similar_to(value, threshold, limit)}[kind]())


def lookup(kind: str, value: str, limit: int = 25, threshold: int = 90,
           client: HttpClient | None = None) -> LookupResult:
    """Look one query up on PubChem and return only structures that pass the screens."""
    curator, accepted, rejected = Curator(), [], Counter()
    for record in candidates(kind, value, limit, threshold, client):
        row, reason = curator.curate(record.smiles)
        if row is None:
            rejected[reason] += 1
        else:
            accepted.append({**row, "source_id": record.source_id, "formula": record.formula})
    return LookupResult(accepted, rejected)
