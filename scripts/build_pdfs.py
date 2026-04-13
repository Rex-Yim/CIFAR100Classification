#!/usr/bin/env python3
"""Build report.pdf and presentation.pdf into artifacts/.

Uses **pdflatex** (two passes) if available; otherwise **tectonic** (e.g. `brew install tectonic`).

Usage (from repo root):
  python scripts/build_pdfs.py

Environment:
  PDFLATEX=/path/to/pdflatex  — optional, force pdflatex binary
  USE_TECTONIC=1  — use tectonic instead of pdflatex (bundles packages; use if BasicTeX is incomplete)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_pdflatex() -> str | None:
    candidates = [
        os.environ.get("PDFLATEX"),
        shutil.which("pdflatex"),
        "/Library/TeX/texbin/pdflatex",
    ]
    for c in candidates:
        if not c:
            continue
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    base = Path("/usr/local/texlive")
    if base.is_dir():
        for pd in sorted(base.glob("*/bin/*/pdflatex")):
            if pd.is_file():
                return str(pd)
    return None


def find_tectonic() -> str | None:
    t = shutil.which("tectonic")
    return t if t and os.access(t, os.X_OK) else None


def compile_pdflatex(pdflatex: str, tex_file: Path, out_dir: Path) -> None:
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


def compile_tectonic(tex_file: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tectonic = find_tectonic()
    if not tectonic:
        raise FileNotFoundError(
            "Neither pdflatex nor tectonic found. Install one of:\n"
            "  brew install --cask basictex   # then add TeX to PATH\n"
            "  brew install tectonic"
        )
    r = subprocess.run(
        [tectonic, "--outdir", str(out_dir), str(tex_file)],
        capture_output=True,
        text=True,
    )
    job = tex_file.stem
    pdf_path = out_dir / f"{job}.pdf"
    if r.returncode != 0:
        print(r.stdout[-6000:] if r.stdout else "", file=sys.stderr)
        print(r.stderr[-2000:] if r.stderr else "", file=sys.stderr)
        raise RuntimeError(f"tectonic failed for {tex_file} with exit {r.returncode}")
    if not pdf_path.is_file():
        raise RuntimeError(f"Expected PDF missing: {pdf_path}")


def compile(tex_file: Path, out_dir: Path, engine: str) -> None:
    if engine == "pdflatex":
        pdflatex = find_pdflatex()
        if not pdflatex:
            raise RuntimeError("pdflatex not available")
        compile_pdflatex(pdflatex, tex_file, out_dir)
    else:
        compile_tectonic(tex_file, out_dir)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts"
    report_tex = root / "report" / "report.tex"
    slides_tex = root / "slides" / "presentation.tex"
    plot_script = root / "scripts" / "plot_training_curves.py"

    for tex in (report_tex, slides_tex):
        if not tex.is_file():
            raise FileNotFoundError(tex)

    if plot_script.is_file():
        r = subprocess.run(
            [sys.executable, str(plot_script)],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stderr[-2000:] if r.stderr else "", file=sys.stderr)
            raise RuntimeError(
                "plot_training_curves.py failed; install matplotlib or fix the log path."
            )
        print(r.stdout.strip() or "Generated training curve figures.")

    use_tectonic = os.environ.get("USE_TECTONIC", "").lower() in ("1", "true", "yes")
    if use_tectonic and find_tectonic():
        engine = "tectonic"
        print("Using: tectonic ->", find_tectonic(), "(USE_TECTONIC=1)")
    elif find_pdflatex():
        engine = "pdflatex"
        print("Using: pdflatex ->", find_pdflatex())
    elif find_tectonic():
        engine = "tectonic"
        print("Using: tectonic ->", find_tectonic())
    else:
        raise FileNotFoundError(
            "No LaTeX engine found. Install pdflatex (BasicTeX/MacTeX) or: brew install tectonic"
        )

    compile(report_tex, artifacts, engine)
    print("Wrote:", artifacts / "report.pdf")

    compile(slides_tex, artifacts, engine)
    print("Wrote:", artifacts / "presentation.pdf")

    if engine == "pdflatex":
        for pat in ("*.aux", "*.log", "*.nav", "*.out", "*.snm", "*.toc"):
            for p in artifacts.glob(pat):
                try:
                    p.unlink()
                except OSError:
                    pass


if __name__ == "__main__":
    main()
