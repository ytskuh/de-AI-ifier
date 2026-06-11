"""Extract text from collected topical-baseline PDFs with MinerU.

Runs $HOME/mineru/bin/mineru (txt method, pipeline backend — arXiv PDFs are
born-digital) over data/baseline/topical/pdfs/, then copies each paper's
markdown to data/baseline/topical/raw/<arxiv_id>.md and records word counts
in the manifest.

Run: uv run python tools/extract_topical_texts.py
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPICAL = ROOT / "data" / "baseline" / "topical"
MINERU = Path(os.environ["HOME"]) / "mineru" / "bin" / "mineru"
WORK = TOPICAL / "mineru_out"


def main():
    manifest_path = TOPICAL / "manifest-topical.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pdfs = [p for p in manifest["papers"] if p.get("pdf_ok")]
    if not pdfs:
        sys.exit("no downloaded PDFs in manifest")

    WORK.mkdir(exist_ok=True)
    raw = TOPICAL / "raw"
    raw.mkdir(exist_ok=True)

    proc = subprocess.run(
        [MINERU, "-p", TOPICAL / "pdfs", "-o", WORK, "-m", "txt", "-b", "pipeline", "-l", "en"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        sys.exit(f"mineru failed:\n{proc.stderr[-3000:]}")

    for p in pdfs:
        stem = Path(p["pdf"]).stem
        candidates = list(WORK.glob(f"{stem}/*/{stem}.md")) or list(WORK.glob(f"{stem}/{stem}.md"))
        if not candidates:
            p["text_ok"] = False
            print(f"  no markdown produced for {stem}")
            continue
        text = candidates[0].read_text(encoding="utf-8")
        (raw / f"{stem}.md").write_text(text, encoding="utf-8")
        p["text"] = f"raw/{stem}.md"
        p["text_ok"] = True
        p["words"] = len(re.findall(r"\S+", text))
        print(f"  {stem}: {p['words']} words")

    ok = [p for p in pdfs if p.get("text_ok")]
    manifest["stats"] = {"extracted_docs": len(ok),
                         "extracted_words": sum(p["words"] for p in ok)}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    print(f"\n{len(ok)}/{len(pdfs)} papers extracted -> {raw}")


if __name__ == "__main__":
    main()
