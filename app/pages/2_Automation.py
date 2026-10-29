"""Readiness, the live model, and the automatic harvest -> retrain -> promote loop."""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import model_stamp  # noqa: E402
from chemgen import DATA_DIR, MODEL_DIR, ROOT  # noqa: E402
from chemgen.automation import Registry, history, read_meta  # noqa: E402
from chemgen.prepare import is_ready, run_checks  # noqa: E402

st.set_page_config(page_title="Automation", layout="wide")
st.title("Automation")
st.caption("The app improves itself: agents harvest new molecules, a candidate generator is retrained, "
           "benchmarked against the live one, and promoted only if it is clearly better.")

PID_FILE, OUT_FILE = MODEL_DIR / "automation.pid", MODEL_DIR / "automation.out"


def running_pid():
    try:
        pid = int(PID_FILE.read_text())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


st.header("1. Is the app prepared?")
checks = run_checks(DATA_DIR, MODEL_DIR)
for c in checks:
    icon = "✅" if c.ok else ("➖" if c.optional else "❌")
    st.markdown(f"{icon} **{c.name}**: {c.detail}" + (f"  \n&nbsp;&nbsp;&nbsp;&nbsp;fix: `{c.fix}`" if not c.ok and c.fix else ""))
if is_ready(checks):
    st.success("Ready. Everything required is installed, downloaded and trained.")
else:
    st.error("Not ready. Run `python -m chemgen prepare` (add `--quick` for a smaller generator, "
             "about 10 minutes), then reload this page.")

st.header("2. Live model")
registry, meta = Registry(MODEL_DIR), read_meta(MODEL_DIR)
st.write(f"Version **{registry.live_version() or 'none'}**"
         + (f", promoted {meta['promoted_at'][:19]}" if meta.get("promoted_at") else "")
         + f". Archived versions: {', '.join(registry.versions()) or 'none'}.")
if meta.get("metrics"):
    m = meta["metrics"]
    cols = st.columns(3)
    cols[0].metric("Useful-molecule yield", f"{m['useful_yield']:.1%}")
    cols[1].metric("Validity", f"{m['validity']:.1%}")
    cols[2].metric("Harvested molecules used", meta.get("harvested_rows", 0))
if st.button("Reload models in this app"):
    st.cache_resource.clear()
    st.rerun()

st.header("3. Run the loop")
pid = running_pid()
if pid:
    st.info(f"A cycle is running (pid {pid}).")
else:
    st.write("A cycle harvests, retrains (about 40 minutes at the default size), benchmarks and promotes. "
             "It runs in the background; this page only starts it and shows the result.")
    if st.button("Start a cycle now"):
        MODEL_DIR.mkdir(exist_ok=True)
        with OUT_FILE.open("w") as out:
            proc = subprocess.Popen([sys.executable, "-m", "chemgen", "automate"], cwd=ROOT, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        PID_FILE.write_text(str(proc.pid))
        st.rerun()
st.markdown("To run it on a schedule, add this line with `crontab -e` (every Sunday at 03:00):")
st.code(f"0 3 * * 0 cd {ROOT} && .venv/bin/python -m chemgen automate >> models/automation.out 2>&1", language="bash")
st.caption("Roll back a bad promotion with `python -m chemgen rollback`.")
if OUT_FILE.exists():
    with st.expander("Latest output"):
        st.code(OUT_FILE.read_text()[-4000:] or "(empty)")

st.header("4. History")
runs = history(MODEL_DIR)
if runs:
    table = pd.DataFrame([{
        "started": r["started"], "new molecules": r["new_since_training"], "retrained": r["retrained"],
        "promoted": r["promoted"], "decision": r["decision"],
        "candidate yield": (r["candidate_metrics"] or {}).get("useful_yield"),
        "live yield": (r["live_metrics"] or {}).get("useful_yield")} for r in reversed(runs)])
    st.dataframe(table, hide_index=True, width="stretch")
else:
    st.info("No cycles have run yet.")
