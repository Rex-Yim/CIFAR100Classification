# Report (LaTeX)

- Source: `report.tex`
- Build PDF (from repository root). The script uses **`pdflatex`** if installed, otherwise **`tectonic`** (`brew install tectonic`):

  ```bash
  python scripts/build_pdfs.py
  ```

  PDFs: `artifacts/report.pdf` and `artifacts/presentation.pdf`.

- Manual build:

  ```bash
  pdflatex -interaction=nonstopmode -output-directory=../artifacts report.tex
  ```

  Run `pdflatex` twice for stable cross-references.
