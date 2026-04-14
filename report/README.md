# Report (LaTeX)

**Course:** MAEG3080 — *Fundamentals of Machine Intelligence* (CUHK). **Repository:** [github.com/Rex-Yim/CIFAR100Classification](https://github.com/Rex-Yim/CIFAR100Classification)

## Sources

| File | Role |
|------|------|
| [`report.tex`](report.tex) | Final project report (PDF) |
| [`../slides/presentation.tex`](../slides/presentation.tex) | Beamer slides |

## Build PDFs (from repository root)

Uses **`pdflatex`** if installed, otherwise **`tectonic`** (`brew install tectonic` on macOS):

```bash
python scripts/build_pdfs.py
```

Output (gitignored): `artifacts/report.pdf` and `artifacts/presentation.pdf`.

## Manual build (from `report/`)

```bash
pdflatex -interaction=nonstopmode -output-directory=../artifacts report.tex
```

Run `pdflatex` twice for stable cross-references and the table of contents.
