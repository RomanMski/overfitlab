# The paper

`main.tex` is the source. Figures come from the package itself, so regenerate
them before compiling or the numbers in the text and the numbers in the plot
can drift apart:

```bash
python scripts/make_figures.py    # from the repository root
tectonic -X compile main.tex      # from this directory
```

A compiled copy is committed as `main.pdf` and mirrored into
`public/paper/overfitlab.pdf`, which is what the site links to.
