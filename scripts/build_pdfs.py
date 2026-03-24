#!/usr/bin/env python3
"""Build report.pdf and presentation.pdf using pdflatex (runs twice per file for stable refs).

Usage (from repo root):
  python scripts/build_pdfs.py

Requires a LaTeX installation with pdflatex (e.g. MacTeX, BasicTeX, or TeX Live).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_pdflatex() -> str:
    candidates = [
        os.environ.get("PDFLATEX"),
        shutil.which("pdflatex"),
        "/Library/TeX/texbin/pdflatex",
        "/usr/local/texlive/*/bin/*/pdflatex",
    ]
    for c in candidates:
        if not c or "*" in c:
            continue
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # glob texlive
    base = Path("/usr/local/texlive")
    if base.is_dir():
        for pd in sorted(base.glob("*/bin/*/pdflatex")):
            if pd.is_file():
                return str(pd)
    raise FileNotFoundError(
        "pdflatex not found. Install LaTeX (e.g. macOS: brew install --cask basictex) "
        "and ensure pdflatex is on PATH, or set PDFLATEX=/path/to/pdflatex."
    )


def compile_tex(pdflatex: str, tex_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    job = tex_file.stem
    env = os.environ.copy()
    env.setdefault("TEXMFOUTPUT", str(out_dir))
    for pass_num in (1, 2):
        r = subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={out_dir}",
                str(tex_file.name),
            ],
            cwd=tex_file.parent,
            env=env,
            capture_output=True,
            text=True,
        )
        pdf_path = out_dir / f"{job}.pdf"
        if r.returncode != 0:
            print(r.stdout[-4000:] if r.stdout else "", file=sys.stderr)
            print(r.stderr[-2000:] if r.stderr else "", file=sys.stderr)
            raise RuntimeError(
                f"pdflatex failed (pass {pass_num}) for {tex_file} with exit {r.returncode}"
            )
        if not pdf_path.is_file():
            raise RuntimeError(f"Expected PDF missing after pass {pass_num}: {pdf_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pdflatex = find_pdflatex()
    print("Using:", pdflatex)

    artifacts = root / "artifacts"
    report_tex = root / "report" / "report.tex"
    slides_tex = root / "slides" / "presentation.tex"

    for tex in (report_tex, slides_tex):
        if not tex.is_file():
            raise FileNotFoundError(tex)

    compile_tex(pdflatex, report_tex, artifacts)
    print("Wrote:", artifacts / "report.pdf")

    compile_tex(pdflatex, slides_tex, artifacts)
    print("Wrote:", artifacts / "presentation.pdf")

    # Clean auxiliary files in artifacts
    for pat in ("*.aux", "*.log", "*.nav", "*.out", "*.snm", "*.toc"):
        for p in artifacts.glob(pat):
            try:
                p.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
