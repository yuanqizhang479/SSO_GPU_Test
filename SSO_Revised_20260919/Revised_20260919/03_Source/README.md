# Matrix-line nuclear-norm research: revised 2026-09-19

This is a separate revision of the 2026-08-04 twelfth manuscript. The original
archive is preserved unchanged. This package contains the English manuscript,
technical supplement, shared LaTeX sources, and CPU mathematical verification.
The accompanying `gpu_suite/` in the outer delivery package is an independent,
optional training study, not the source of the numerical tables in this paper.

## Read and rebuild

The delivered main and supplement PDFs use SIAM's dated `siamart251216` class.
Author and institution information has not been invented. A main paper over
SIAM's usual 20-page policy requires an editorial length decision; successful
compilation does not establish submission readiness.

Two builds use exactly the same section files:

```bash
python build_pdfs.py --format review
# Produces submission/main_review.pdf and submission/supplement_review.pdf.

python download_siam_template.py
python build_pdfs.py --format siam
# Or: python build_pdfs.py --format siam --template-dir /path/to/official/bundle
```

Install a normal TeX Live distribution with latex-extra, recommended fonts and
science packages (or equivalent MiKTeX packages). The official SIAM macro server
returned HTTP 403 during this preparation. The official class was obtained for
local compilation from an identified public academic source archive; its
redistribution terms prohibit shipping the class alone without its complete
bundle, so an incomplete bundle is not distributed here. The download script
retrieves the complete official bundle when that server is accessible. The
portable article build is self-contained apart from normal TeX dependencies;
it has different formatting/pagination and is not presented as official SIAM
format. See `TEMPLATE_PROVENANCE.md` for the exact dependency provenance.

Build the main document before the supplement because external references
use its auxiliary file. `build_pdfs.py` does this in order and checks unresolved
citations/references. The generated TeX table file belongs with the source.

## Reproduce the mathematical examples

```bash
python -m pip install -r requirements.txt
python verify_all.py
```

The suite runs the original ten scientific checks, a high-precision publication
table generator/check, the actual-output certificate example, and the finite
cancellation criterion and nonvanishing boundary-direction coefficient. It records source/output hashes, environment and logs
under `results/`. A failing check exits nonzero. `python -O verify_all.py` and
`PYTHONOPTIMIZE=1` are deliberately rejected so scientific assertions cannot
silently disappear. New checks using explicit exceptions remain active too.

Regenerate display values only after an intended numerical change:

```bash
python verify_publication_tables.py --help
python verify_publication_tables.py
python verify_publication_tables.py --check
python verify_all.py
```

The precise supported flags are documented by that script; the checked-in JSON
and TeX are the reference snapshot. High-precision mpmath/Decimal computations
are numerical evidence, not formal proofs or interval-certified enclosures.
Finite-difference checks of intrinsic constants remain finite-difference checks;
they are not called symbolic differentiation.

## Scope

The main results concern prescribed matrix lines. Strict-crossing laws permit
arbitrary rectangular shapes and coranks; the boundary/crossover results retain
their stated square, transverse corank-one and positive-curvature assumptions.
The SSO rank-one normal is a special case; hard and pseudo-Huber centers agree
there under the stated conditions. Neither a full-space open failure set nor
synthetic examples estimate occurrence rates in training.

Actual returned directions, approximate/exact normals, substituted compact-polar
directions, feasibility residuals and repaired primal-dual gaps are distinct
quantities. Ordinary floating-point evaluation of an exact-real inequality is
not a machine-rigorous certificate. The finite-NS code in the separate GPU study
is not automatically covered by the Fenchel smoothing-family theorem.

No GPU training result is claimed by this source package. See the outer
Chinese revision decision, research route and validation report for changes,
limitations, and the relation to Xie et al., arXiv:2601.08393v3.
