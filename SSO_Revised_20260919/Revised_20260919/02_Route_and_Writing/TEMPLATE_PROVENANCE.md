# Template provenance and build instructions

Checked 2026-09-19. The SIAM author resource page currently lists `siamart251216.cls`, version 2025-12-16 v1.4.8. Official sources:

- https://epubs.siam.org/journal-authors#macros
- https://epubs.siam.org/pb-assets/macros/standard/siamart_251216.zip
- https://epubs.siam.org/pb-assets/macros/standard/docsiamart.pdf
- https://epubs.siam.org/journal/simax/instructions-for-authors

The official complete ZIP and individual downloads returned HTTP 403 in the preparation environment. For local format checking only, two unmodified files with the current official names/version were obtained from the publicly available source archive https://arxiv.org/src/2602.09198v1 . They were stored outside this manuscript source archive; this is a public mirror source, not a claim that an official-bundle byte comparison succeeded.

| Local compilation dependency | SHA-256 |
|---|---|
| `siamart251216.cls` | `33f2eb091c8bcbda9baed2e868489460da868047b0a518b4ce1b2f2c15fda530` |
| `siamplain.bst` | `a5df7c482dc7d4459f08b91c28437c3779284dcd82591acf1c977de7508ac6c7` |

The class header explicitly requires distribution together with the complete SIAM macro distribution and prohibits distributing the class alone. We therefore do not redistribute these isolated files. No third-party class was edited. The user-facing source is buildable without these files through the portable review path below. A complete official distribution is needed to reproduce the SIAM layout.

## SIAM layout

Run:

```bash
python download_siam_template.py
python build_pdfs.py --format siam
```

If automated download is blocked, download the **complete standard macro ZIP** from the official author page and import it:

```bash
python download_siam_template.py --zip /path/to/siamart_251216.zip
python build_pdfs.py --format siam
```

The installer checks the eight required named files, preserves the whole ZIP contents without altering the macros, and records bundle provenance. An already installed distribution is supported with `python build_pdfs.py --format siam --template-dir /path/to/template`.

## Portable review layout

```bash
python build_pdfs.py --format review
```

This requires no downloaded SIAM class. It uses `submission/main_review.tex` and `submission/supplement_review.tex`, which input the same manuscript bodies. Outputs are `main_review.pdf` and `supplement_review.pdf`. Their page count, bibliography layout, and equation numbering are not the SIAM page count/layout.

## TeX dependencies

Install an up-to-date TeX Live or MiKTeX with amsmath, mathtools, ntheorem, algorithm, float, cleveref, xr-hyper, hyperref, lmodern, microtype, enumitem, booktabs, and longtable. Typical Debian packages are texlive-latex-extra, texlive-fonts-recommended, and texlive-science. The preparation environment lacked algorithm.sty. The complete public CTAN algorithms distribution was downloaded from https://mirrors.ibiblio.org/CTAN/macros/latex/contrib/algorithms.zip and its standard `latex algorithms.ins` installation was used outside this archive; its license and upstream files were retained locally. It is a normal TeX dependency, not a custom replacement class or an omitted research source file.

## Publication status and formatting limits

The manuscript retains anonymous metadata because verified author names/affiliations were not supplied in this task. Replace these before an actual identified submission. The main entry uses the official class without the line-numbered `review` option for a clean reading PDF; adding the SIAM `review` option is available for a journal submission. This is distinct from our `--format review`, which means portable article layout.

SIMAX states a 20-page policy but can consider longer papers. The current official-class PDF may exceed 20 pages; its actual count is reported in the final build results. Using the class does not itself establish submission readiness. The final source archive has not been submitted to any journal or public repository.
