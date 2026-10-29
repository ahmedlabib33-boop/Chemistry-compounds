import http.server
import threading
import urllib.request
from collections import Counter

import pandas as pd
import pytest

from chemgen import harvest
from chemgen.harvest import (AgentReport, ChEMBLSource, Curator, HttpClient, HttpError, PubChemSource,
                             RawRecord, Refused, SourceAgent, Store, lookup)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
CAFFEINE_LIKE = "CCNC(=O)c1ccc(F)cc1"


# ---------- HTTP client against a local server ----------

class _Handler(http.server.BaseHTTPRequestHandler):
    hits: list = []
    failures_left = 0

    def log_message(self, *args):
        pass

    def _reply(self):
        type(self).hits.append((self.command, self.path, self.rfile.read(int(self.headers.get("Content-Length", 0)))))
        if self.path.startswith("/missing"):
            self.send_response(404); self.end_headers(); return
        if self.path.startswith("/flaky") and type(self).failures_left > 0:
            type(self).failures_left -= 1
            self.send_response(503); self.send_header("Retry-After", "0"); self.end_headers(); return
        if self.path.startswith("/broken"):
            self.send_response(500); self.end_headers(); return
        body = b'{"ok": true}'
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _reply


@pytest.fixture()
def server():
    _Handler.hits, _Handler.failures_left = [], 0
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def client(tmp_path=None, **kw):
    sleeps = []
    c = HttpClient(opener=urllib.request.build_opener(urllib.request.ProxyHandler({})),
                   cache_dir=tmp_path, sleep=sleeps.append, **kw)
    c.sleeps = sleeps
    return c


def test_client_returns_json_and_none_for_404(server):
    c = client()
    assert c.get_json(f"{server}/ok") == {"ok": True}
    assert c.get_json(f"{server}/missing") is None


def test_client_retries_transient_errors_then_succeeds(server):
    _Handler.failures_left = 2
    c = client()
    assert c.get_json(f"{server}/flaky") == {"ok": True}
    assert len(_Handler.hits) == 3


def test_client_raises_after_retries_are_exhausted(server):
    with pytest.raises(HttpError, match="HTTP 500"):
        client(retries=2).get_json(f"{server}/broken")
    assert len(_Handler.hits) == 3  # first try plus two retries


def test_client_posts_form_data_and_caches(server, tmp_path):
    c = client(tmp_path / "cache")
    for _ in range(2):
        assert c.get_json(f"{server}/ok", data={"cid": "1,2,3"}) == {"ok": True}
    assert len(_Handler.hits) == 1                      # second call served from disk
    assert _Handler.hits[0][0] == "POST" and _Handler.hits[0][2] == b"cid=1%2C2%2C3"


def test_client_throttles_between_requests(server):
    c = client(min_interval=5.0)
    c.get_json(f"{server}/a")
    c.get_json(f"{server}/b")
    assert c.sleeps and 0 < max(c.sleeps) <= 5.0


# ---------- fake client for the sources ----------

class FakeClient:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def _sleep(self, seconds):
        pass

    def get_json(self, url, data=None):
        self.calls.append((url, data))
        for needle, payload in self.routes:
            if needle in url:
                return payload(url, data) if callable(payload) else payload
        return None


def props(*pairs):
    return {"PropertyTable": {"Properties": [{"CID": cid, "SMILES": smi, "MolecularFormula": "X"} for cid, smi in pairs]}}


def test_pubchem_random_stream_parses_batches():
    fake = FakeClient([("compound/cid/property", props((1, ASPIRIN), (2, IBUPROFEN)))])
    src = PubChemSource(fake, batch=5)
    stream = src.stream()
    first, second = next(stream), next(stream)
    assert (first.smiles, first.source, first.source_id) == (ASPIRIN, "pubchem", "1") and second.source_id == "2"
    url, data = fake.calls[0]
    assert len(data["cid"].split(",")) == 5 and "SMILES" in url


def test_pubchem_formula_waits_for_asynchronous_listkey():
    fake = FakeClient([
        ("fastformula", {"Waiting": {"ListKey": "abc123"}}),
        ("listkey/abc123", {"IdentifierList": {"CID": [2244]}}),
        ("compound/cid/property", props((2244, ASPIRIN))),
    ])
    records = list(PubChemSource(fake).by_formula("C9H8O4", 10))
    assert [r.source_id for r in records] == ["2244"]


def test_pubchem_similarity_posts_smiles():
    fake = FakeClient([("fastsimilarity_2d", {"IdentifierList": {"CID": [2244]}}),
                       ("compound/cid/property", props((2244, ASPIRIN)))])
    list(PubChemSource(fake).similar_to(ASPIRIN, 95, 5))
    assert fake.calls[0][1] == {"smiles": ASPIRIN} and "Threshold=95" in fake.calls[0][0]


def test_pubchem_missing_results_are_empty_not_errors():
    assert list(PubChemSource(FakeClient([])).by_name("no-such-thing")) == []


def test_chembl_pages_until_there_is_no_next_link():
    def page(url, data):
        if "limit=1&" in url:
            return {"page_meta": {"total_count": 4}, "molecules": []}
        offset = int(url.split("offset=")[1].split("&")[0])
        mols = [{"molecule_chembl_id": f"CHEMBL{offset + i}", "molecule_structures": {"canonical_smiles": ASPIRIN}}
                for i in range(2)] + [{"molecule_chembl_id": "NOSTRUCT", "molecule_structures": None}]
        return {"page_meta": {"next": "more" if offset < 2 else None}, "molecules": mols}

    src = ChEMBLSource(FakeClient([("molecule.json", page)]), page_size=2)
    ids = [r.source_id for r in src.stream()]
    assert ids and "NOSTRUCT" not in ids and all(i.startswith("CHEMBL") for i in ids)


# ---------- curation ----------

def test_curator_standardises_and_accepts_a_drug_like_record():
    row, reason = Curator().curate(f"{ASPIRIN}.[Na+].[Cl-]")  # salt/counter-ions stripped
    assert reason == "ok" and row["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"
    assert {"logP", "qed", "SAS"} <= set(row)


@pytest.mark.parametrize("smiles, reason", [
    ("not a smiles", "unparseable"),
    ("C" * 85, "too long"),
    ("CCOP(=O)(OCC)c1ccccc1", "disallowed elements"),
    ("CC(C)(C)c1ccc(CN=[N+]=[N-])cc1", "hazard screen"),
    ("S=C1SC(=Cc2ccccc2)C(=O)N1", "structural alert"),
])
def test_curator_rejections_use_stable_reason_keys(smiles, reason):
    row, got = Curator().curate(smiles)
    assert row is None and got == reason


def test_curator_neutralises_charges():
    row, _ = Curator().curate("COc1ccc([C@H]([NH3+])c2ccc(F)cc2)cc1")
    assert row is not None and "+" not in row["smiles"]


# ---------- store ----------

def test_store_dedupes_persists_and_reloads(tmp_path):
    path = tmp_path / "out" / "h.csv"
    store = Store(path)
    row = dict(smiles=ASPIRIN, logP=1.3, qed=0.55, SAS=1.6, source="pubchem", source_id="1", fetched_at="2026-01-01")
    assert store.add(row) and not store.add(dict(row, source_id="2"))
    assert len(Store(path)) == 1 and not Store(path).add(row)
    assert list(pd.read_csv(path).columns) == Store.FIELDS


def test_store_is_safe_under_concurrent_adds(tmp_path):
    store = Store(tmp_path / "h.csv")
    row = dict(smiles=ASPIRIN, logP=1.3, qed=0.55, SAS=1.6, source="x", source_id="1", fetched_at="d")
    results = []
    threads = [threading.Thread(target=lambda: results.append(store.add(dict(row)))) for _ in range(8)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert results.count(True) == 1 and len(pd.read_csv(tmp_path / "h.csv")) == 1


# ---------- agents ----------

class ListSource:
    name = "fake"

    def __init__(self, smiles, then_fail=False):
        self.smiles, self.then_fail = smiles, then_fail

    def stream(self):
        for i, s in enumerate(self.smiles):
            yield RawRecord(s, self.name, str(i))
        if self.then_fail:
            raise HttpError("network down")


def run_agent(tmp_path, source, budget=10, **kw):
    store = Store(tmp_path / "h.csv")
    return SourceAgent(source, Curator(), store, budget, **kw).run(), store


def test_agent_counts_accepted_duplicates_and_rejections(tmp_path):
    report, store = run_agent(tmp_path, ListSource([ASPIRIN, ASPIRIN, IBUPROFEN, "CCOP(=O)(OCC)c1ccccc1", "nonsense"]))
    assert (report.fetched, report.accepted, report.duplicates) == (5, 2, 1)
    assert report.rejected == Counter({"disallowed elements": 1, "unparseable": 1}) and len(store) == 2


def test_agent_stops_at_budget(tmp_path):
    report, _ = run_agent(tmp_path, ListSource([ASPIRIN, IBUPROFEN, CAFFEINE_LIKE]), budget=2)
    assert report.accepted == 2 and report.fetched == 2


def test_agent_stops_when_a_source_yields_too_little(tmp_path):
    report, _ = run_agent(tmp_path, ListSource(["nonsense"] * 50), budget=5, max_fetched=7)
    assert report.fetched == 7 and report.accepted == 0


def test_network_failure_ends_that_agent_with_its_partial_results(tmp_path):
    report, store = run_agent(tmp_path, ListSource([ASPIRIN], then_fail=True))
    assert report.error == "network down" and report.accepted == 1 and len(store) == 1
    assert "STOPPED" in report.line()


def test_coordinator_runs_every_source_into_one_store(tmp_path, monkeypatch):
    class A(ListSource):
        def __init__(self, client, rng=None):
            super().__init__([ASPIRIN, IBUPROFEN]); self.name = "a"

    class B(ListSource):
        def __init__(self, client, rng=None):
            super().__init__([IBUPROFEN, CAFFEINE_LIKE]); self.name = "b"

    monkeypatch.setattr(harvest, "SOURCES", {"a": (A, 0.0), "b": (B, 0.0)})
    logs = []
    reports = harvest.harvest(["a", "b"], budget=5, out_path=tmp_path / "h.csv", log=logs.append)
    assert sum(r.accepted for r in reports) == 3          # ibuprofen is only stored once
    assert sum(r.duplicates for r in reports) == 1
    assert len(pd.read_csv(tmp_path / "h.csv")) == 3 and "3 molecules in total" in logs[-1]


# ---------- lookup and request screening ----------

def formula_client():
    return FakeClient([
        ("fastformula", {"IdentifierList": {"CID": [2244, 1, 2]}}),
        ("compound/cid/property", props((2244, ASPIRIN), (1, "CCOP(=O)(OCC)c1ccccc1"), (2, "nonsense"))),
    ])


def test_lookup_returns_only_structures_that_pass_the_screens():
    result = lookup("formula", "C9H8O4", client=formula_client())
    assert [r["smiles"] for r in result.accepted] == [ASPIRIN]
    assert result.rejected == Counter({"disallowed elements": 1, "unparseable": 1})


@pytest.mark.parametrize("bad", ["aspirin", "C9H8O4!", "c9h8o4", "", "C9H8O4 OR 1=1"])
def test_lookup_rejects_malformed_formulas(bad):
    with pytest.raises(ValueError):
        lookup("formula", bad, client=FakeClient([]))


@pytest.mark.parametrize("kind, value", [("name", "sarin"), ("name", "a nerve agent"), ("name", "fentanyl"),
                                         ("similar", "CC(N)Cc1ccccc1")])
def test_lookup_refuses_harmful_queries_before_any_request(kind, value):
    fake = FakeClient([])
    with pytest.raises(Refused):
        lookup(kind, value, client=fake)
    assert fake.calls == []


def test_lookup_by_name_works_for_ordinary_compounds():
    fake = FakeClient([("compound/name/aspirin", props((2244, ASPIRIN)))])
    assert [r["source_id"] for r in lookup("name", "aspirin", client=fake).accepted] == ["2244"]


def test_agent_report_line_is_readable():
    r = AgentReport("pubchem", fetched=10, accepted=7, duplicates=1, rejected=Counter({"structural alert": 2}), seconds=3)
    assert "accepted 7" in r.line() and "structural alert 2" in r.line()
