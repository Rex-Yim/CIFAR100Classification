# Report (LaTeX)

- Source: `report.tex`
- Build PDF (from repository root, needs `pdflatex` on your PATH):

  ```bash
  python scripts/build_pdfs.py
  ```

  PDF is written to `artifacts/report.pdf`.

- Manual build:

  ```bash
  pdflatex -interaction=nonstopmode -output-directory=../artifacts report.tex
  ```

  Run `pdflatex` twice for stable cross-references.
