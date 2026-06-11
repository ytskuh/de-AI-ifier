"""Lexical layer: run Vale (ai-tells + Deslop + Deaiify styles) and map alerts to Findings."""

import json
import shutil
import subprocess
from pathlib import Path

from .findings import SEVERITY_WEIGHT, Finding

ROOT = Path(__file__).resolve().parents[2]
VALE = ROOT / "bin" / "vale"
VALE_INI = ROOT / "rulepacks" / "en" / "vale.ini"
STYLES = ROOT / ".vale-styles"


def ensure_styles() -> None:
    """Sync remote packages once; refresh the generated Deaiify style every run."""
    if not (STYLES / "ai-tells").exists():
        subprocess.run([VALE, "--config", VALE_INI, "sync"], check=True, capture_output=True)
    src = ROOT / "rulepacks" / "en" / "Deaiify"
    dst = STYLES / "Deaiify"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def run(path: Path) -> list[Finding]:
    ensure_styles()
    proc = subprocess.run(
        [VALE, "--config", VALE_INI, "--output=JSON", "--no-exit", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"vale failed: {proc.stderr[:500]}")
    alerts = json.loads(proc.stdout or "{}")
    findings = []
    for file_alerts in alerts.values():
        for a in file_alerts:
            action = a.get("Action") or {}
            findings.append(Finding(
                line=a["Line"],
                span=tuple(a["Span"]),
                layer="lexical",
                rule=a["Check"],
                severity=SEVERITY_WEIGHT.get(a["Severity"], 0.3),
                message=a["Message"],
                match=a["Match"],
                payload={"action": action} if action.get("Name") else {},
            ))
    return findings
