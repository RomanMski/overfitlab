# StressFold technical paper

The paper is authored in `main.tex`. All numerical figures and tables are
generated from fixed-seed controlled experiments, and they do not call the
StressFold package API.

A compiled and visually checked copy is committed as `main.pdf`.

From the repository root:

```powershell
py scripts/reproduce_paper.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Tectonic can be used instead of a full TeX Live installation:

```powershell
tectonic --keep-logs main.tex
```

The reproduction script requires NumPy, SciPy, Matplotlib, and scikit-learn.
The PDF build requires either Tectonic or a LaTeX distribution with `latexmk`,
`pdflatex`, BibTeX, and the packages declared at the top of `main.tex`.

Generated artifacts are written to `paper/figures/` and `paper/tables/`. The
CSV table is retained beside its LaTeX rendering to make the reported values
easy to audit.
