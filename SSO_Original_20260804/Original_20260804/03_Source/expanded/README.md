# SIMAX-target twelfth revision package

This package contains the twelfth, theory-expanded revision dated August 4,
2026.

## Manuscripts

1. `submission/main.tex` and `submission/main.pdf`: the 22-page main paper,
   **Nuclear-Norm Best Approximation at Rank-Deficient Matrix-Line
   Crossings: Certificate Geometry and Singular Smoothing**.
2. `submission/supplement.tex` and `submission/supplement.pdf`: the 11-page
   technical supplement.
3. `SUPPLEMENTARY_MATERIALS_INDEX.html`: the SIAM-format index for the
   supplementary PDF and reproducibility archive.

The PDFs use a compact anonymous A4 review layout. Before journal upload,
recompile in the then-current official SIAM class and recheck the page count,
running title, author metadata, and front matter. The present 22 pages do not
certify compliance under a different class file.

Compile from `submission/`, in this order:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The supplement imports cross-references from `main.aux`, so compile the main
paper first and do not run the two builds concurrently.

## Deterministic verification

Run the complete suite from the package root:

```bash
python3 -m pip install -r requirements.txt
python3 verify_all.py
```

The following extended-precision checks use Python's standard library:

```bash
python3 verify_rate_constants_exact.py
python3 verify_unified_crossover.py
python3 verify_corank2_strict_center.py
```

The matrix checks use NumPy and SciPy:

```bash
python3 verify_partial_isometry.py
python3 verify_direction_rates.py
python3 verify_theorems.py
python3 verify_arbitrary_corank_strict_law.py
python3 verify_direction_rate_general_corank.py
python3 verify_intrinsic_constants.py
```

The smoothing-family and fifth-order checks additionally use mpmath:

```bash
python3 verify_smoothing_family.py
```

`verify_all.py` executes ten assertion-based scripts and exits nonzero on any
failure. The suite covers tall and wide shapes, unequal nullities,
rank-deficient kernel compression, repeated positive singular values,
arbitrary corank, non-diagonal direction coefficients, Huber multiplier
nonuniqueness, polynomial and finite-saturation tails, and the sharp
cubic/quadratic and quintic/quartic compact-root branches. The legacy
`verify_crossover_constants.py` entry point is a compatibility wrapper and is
not an additional experiment.

The suite was last run successfully on August 4, 2026 with Python 3.12.13,
NumPy 2.3.5, SciPy 1.17.0, and mpmath 1.3.0. The PDFs were compiled with
latexmk 4.83 and pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023/Debian).
