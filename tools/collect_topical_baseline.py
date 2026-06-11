"""Collect a topical pre-LLM baseline corpus by seeded random sampling from arXiv.

Anti-bias protocol (every choice declared, no model-chosen papers):
- POPULATION: arXiv papers whose PRIMARY category is one of CATEGORIES, first
  submitted in [WINDOW_START, WINDOW_END] (strictly pre-ChatGPT), and carrying
  a journal-ref or DOI (peer-reviewed/published proxy = the quality filter).
- ENUMERATION: arXiv API listing sorted by submittedDate ascending — a stable,
  relevance-free ordering of the whole population.
- SAMPLING: uniform random offsets from a seeded RNG, rejection-sampled against
  the declared filters until K papers per category are accepted. No keyword
  queries, no relevance ranking, no per-paper judgment anywhere.
- Reproducible from SEED; protocol + every accept/reject recorded in the manifest.

Run: uv run python tools/collect_topical_baseline.py
"""

import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "baseline" / "topical"

# Target paper sits at the intersection of numerical analysis for SDEs,
# applied probability, computational statistics, and ML theory.
CATEGORIES = ["math.NA", "math.PR", "stat.CO", "stat.ML"]
K_PER_CATEGORY = 6
WINDOW_START, WINDOW_END = "20180101000000", "20221031235959"
SEED = 20260610
MAX_ATTEMPTS_FACTOR = 8  # give up on a category after K * this rejections
# arXiv API 500s on start offsets beyond ~10k, so the window is partitioned
# into half-year slices (each far below the limit); a global uniform draw maps
# to (slice, in-slice offset) — same distribution, shallow pagination only.
SLICES = [("20180101000000", "20180630235959"), ("20180701000000", "20181231235959"),
          ("20190101000000", "20190630235959"), ("20190701000000", "20191231235959"),
          ("20200101000000", "20200630235959"), ("20200701000000", "20201231235959"),
          ("20210101000000", "20210630235959"), ("20210701000000", "20211231235959"),
          ("20220101000000", "20220630235959"), ("20220701000000", "20221031235959")]
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
SLEEP = 3.0  # arXiv API etiquette


def api_get(params: dict) -> ET.Element:
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                tree = ET.fromstring(r.read())
            time.sleep(SLEEP)
            return tree
        except Exception as e:
            last = e
            time.sleep(SLEEP * (attempt + 2))
    raise RuntimeError(f"arXiv API failed after retries: {url}") from last


def query_str(cat: str, lo: str, hi: str) -> str:
    return f"cat:{cat} AND submittedDate:[{lo} TO {hi}]"


def slice_totals(cat: str) -> list[int]:
    totals = []
    for lo, hi in SLICES:
        tree = api_get({"search_query": query_str(cat, lo, hi), "start": 0, "max_results": 1})
        totals.append(int(tree.find("os:totalResults", NS).text))
    return totals


def entry_meta(e: ET.Element) -> dict:
    get = lambda tag, ns="a": (e.find(f"{ns}:{tag}", NS).text or "").strip() if e.find(f"{ns}:{tag}", NS) is not None else ""
    arxiv_id = get("id").rsplit("/", 1)[-1]
    return {
        "arxiv_id": arxiv_id,
        "title": " ".join(get("title").split()),
        "published": get("published"),
        "primary_category": e.find("arxiv:primary_category", NS).attrib.get("term", ""),
        "journal_ref": get("journal_ref", "arxiv"),
        "doi": get("doi", "arxiv"),
    }


def sample_category(cat: str) -> tuple[list[dict], dict]:
    totals = slice_totals(cat)
    total = sum(totals)
    rng = random.Random(f"{SEED}:{cat}")
    draws = rng.sample(range(total), min(total, K_PER_CATEGORY * MAX_ATTEMPTS_FACTOR))
    accepted, rejected = [], []
    for global_off in draws:
        if len(accepted) >= K_PER_CATEGORY:
            break
        s, off = 0, global_off
        while off >= totals[s]:
            off -= totals[s]
            s += 1
        lo, hi = SLICES[s]
        tree = api_get({"search_query": query_str(cat, lo, hi), "start": off, "max_results": 1,
                        "sortBy": "submittedDate", "sortOrder": "ascending"})
        entries = tree.findall("a:entry", NS)
        if not entries:
            rejected.append({"offset": global_off, "reason": "empty_page"})
            continue
        m = entry_meta(entries[0])
        m["offset"] = global_off
        if m["primary_category"] != cat:
            rejected.append({**m, "reason": f"primary_category={m['primary_category']}"})
        elif not (m["journal_ref"] or m["doi"]):
            rejected.append({**m, "reason": "no journal_ref/doi"})
        else:
            accepted.append(m)
            print(f"  [{cat}] {len(accepted)}/{K_PER_CATEGORY} {m['arxiv_id']}  {m['title'][:60]}")
    return accepted, {"category": cat, "population_size": total,
                      "accepted": len(accepted), "rejected": len(rejected),
                      "rejections": rejected}


def download_pdf(arxiv_id: str, dest: Path) -> bool:
    url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            time.sleep(10)  # arXiv rate-limits PDF fetches harder than the API
            if data.startswith(b"%PDF"):
                dest.write_bytes(data)
                return True
            return False
        except urllib.error.HTTPError as e:
            if e.code == 429:  # back off hard and retry
                time.sleep(60 * (attempt + 1))
                continue
            print(f"  download failed {arxiv_id}: {e}")
            time.sleep(10)
            return False
        except Exception as e:
            print(f"  download failed {arxiv_id}: {e}")
            time.sleep(10)
            return False
    print(f"  download failed {arxiv_id}: rate-limited after retries")
    return False


def download_bytes(url: str) -> bytes | None:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            time.sleep(15)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60 * (attempt + 1))
                continue
            print(f"  fetch failed {url}: {e}")
            time.sleep(15)
            return None
        except Exception as e:  # IncompleteRead etc. — retry
            print(f"  fetch retry {url}: {e}")
            time.sleep(30)
    return None


def flatten_tex(main: Path, depth: int = 3) -> str:
    """Inline \\input/\\include one file deep at a time (best effort)."""
    text = main.read_text(encoding="utf-8", errors="replace")
    if depth == 0:
        return text

    def repl(m):
        name = m.group(1)
        for cand in (main.parent / name, main.parent / f"{name}.tex"):
            if cand.is_file():
                return flatten_tex(cand, depth - 1)
        return m.group(0)

    return re.sub(r"\\(?:input|include)\{([^}]+)\}", repl, text)


def fetch_sources():
    """Download arXiv e-print sources for accepted papers -> latex/<id>.tex."""
    import gzip
    import io
    import tarfile
    import tempfile

    path = OUT / "manifest-topical.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    latex_dir = OUT / "latex"
    latex_dir.mkdir(exist_ok=True)
    for m in manifest["papers"]:
        if m.get("latex_ok"):
            continue
        aid = m["arxiv_id"]
        data = download_bytes(f"https://export.arxiv.org/e-print/{aid}")
        if data is None:
            m["latex_ok"] = False
            continue
        stem = aid.replace("/", "_")
        tex = None
        if data.startswith(b"%PDF"):
            m["latex_ok"] = False
            m["latex_note"] = "pdf-only submission"
            print(f"  {aid}: pdf-only, no source")
            continue
        try:  # tarball of a multi-file project
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar, \
                 tempfile.TemporaryDirectory() as td:
                tar.extractall(td, filter="data")
                texs = list(Path(td).rglob("*.tex"))
                mains = [t for t in texs if "\\begin{document}" in
                         t.read_text(encoding="utf-8", errors="replace")]
                if mains:
                    main = max(mains, key=lambda t: t.stat().st_size)
                    tex = flatten_tex(main)
        except tarfile.TarError:
            try:  # gzipped single .tex
                cand = gzip.decompress(data).decode("utf-8", errors="replace")
                if "\\begin{document}" in cand:
                    tex = cand
            except OSError:
                pass
        if tex:
            (latex_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            m["latex"] = f"latex/{stem}.tex"
            m["latex_ok"] = True
            m["latex_words"] = len(re.findall(r"\S+", tex))
            print(f"  {aid}: {m['latex_words']} words (raw tex)")
        else:
            m["latex_ok"] = False
            print(f"  {aid}: no main .tex found")
    ok = sum(1 for p in manifest["papers"] if p.get("latex_ok"))
    manifest.setdefault("stats", {})["latex_docs"] = ok
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{ok}/{len(manifest['papers'])} latex sources -> {latex_dir}")


def retry_downloads():
    """Re-attempt failed PDF downloads recorded in the manifest."""
    path = OUT / "manifest-topical.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = [m for m in manifest["papers"] if not m.get("pdf_ok")]
    print(f"retrying {len(missing)} downloads …")
    for m in missing:
        m["pdf_ok"] = download_pdf(m["arxiv_id"], OUT / m["pdf"])
        print(f"  {m['arxiv_id']}: {'ok' if m['pdf_ok'] else 'FAILED'}")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = sum(1 for m in manifest["papers"] if m.get("pdf_ok"))
    print(f"{ok}/{len(manifest['papers'])} PDFs present")


def main():
    pdf_dir = OUT / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    all_accepted, stats = [], []
    for cat in CATEGORIES:
        print(f"sampling {cat} …")
        acc, st = sample_category(cat)
        all_accepted += acc
        stats.append(st)

    print("downloading PDFs …")
    for m in all_accepted:
        fname = m["arxiv_id"].replace("/", "_") + ".pdf"
        m["pdf"] = f"pdfs/{fname}"
        m["pdf_ok"] = download_pdf(m["arxiv_id"], pdf_dir / fname)

    manifest = {
        "name": "topical-baseline-arxiv",
        "protocol": {
            "description": "Seeded uniform random sampling from an enumerated population; "
                           "no keyword search, no relevance ranking, no model-chosen papers. "
                           "Quality filter: journal_ref or DOI present (published proxy). "
                           "Pre-LLM filter: first submission before 2022-11 (ChatGPT release).",
            "categories": CATEGORIES, "k_per_category": K_PER_CATEGORY,
            "window": [WINDOW_START, WINDOW_END], "seed": SEED,
            "enumeration": "arXiv API, sortBy=submittedDate ascending, half-year slices "
                           "(uniform global draw mapped to slice+offset; API 500s past ~10k offsets)",
            "caveat": "population_size snapshots drift slightly as arXiv metadata changes; "
                      "rerun fidelity requires same-day execution or the recorded id list.",
        },
        "license_note": "PDFs for local corpus use only — do not redistribute.",
        "papers": all_accepted,
        "sampling_stats": stats,
    }
    (OUT / "manifest-topical.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = sum(1 for m in all_accepted if m.get("pdf_ok"))
    print(f"\n{len(all_accepted)} papers accepted, {ok} PDFs downloaded -> {OUT}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "retry":
        retry_downloads()
    elif len(sys.argv) > 1 and sys.argv[1] == "sources":
        fetch_sources()
    else:
        main()
