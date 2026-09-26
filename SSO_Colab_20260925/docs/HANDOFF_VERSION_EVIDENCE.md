# Versions and Evidence Boundaries

## 1. Which file is which version

| Version | Content obtained this time | Evidence status |
|---|---|---|
| 2026-08-04, 12th version | Manuscript, supplement, source code, and reports from that time, contained in the 8/4 original ZIP | Recovered; the separately found 12th-version source has the same hash as the source inside that ZIP |
| 2026-09-13, 13th version | Completion report, formulas, and some historical file SHAs from this conversation | Neither the exact PDF nor the source ZIP was recovered; the historical numbers below must not be treated as run results of the current source |
| 2026-09-19, independent revision | Full revision ZIP, extracted files, standalone GPU ZIP | Recovered; key documents and implementation actually read; the in-package report explicitly says it was redone from 8/4 and cannot be assumed to inherit 9/13 |
| 2026-09-25, this handoff | Added context, version and missing-item tables, index, archive checksums | Migration and inventory only; no new research theorems, no changes to training algorithms, no scientific experiments run |

## 2. The numbers and algorithms most easily confused

| Item | 9/13 conversation report | 9/19 actually recovered archive |
|---|---|---|
| Paper page count | article main text 23, supplement 11 | SIAM class 30/15; portable article version 26/13 |
| Math verification | Original ten groups + 69 numerically generated records | 14 items; see verification_latest.json and logs |
| GPU direction rules | compact, pseudo-Huber, certified reference solver | AdamW, Muon, spherical Muon, SSO with finite NS, finite NS + SVD repair |
| tokenizer | GPT-2/tiktoken | OLMo-2 tokenizer |
| Data plan | FineWeb as main, TinyStories for debugging, OLMo later | TinyStories, OLMo seven sources, FineWeb-Edu |
| Micro real-data run | train 4096 / val 1024 tokens, 3 CPU steps | train 2147 / val 624 / test 668 tokens, 4 CPU steps |
| Actual GPU training | None | Historical status explicitly false; not run this round either |

This does not declare 9/19 a lossless upgrade of 9/13; it is meant to give a new chat a real, readable, executable starting point. The 9/13 conclusions must be compared item by item and must not be silently dropped or given a fabricated inheritance relationship.

## 3. Where the key 9/13 changes correspond in 9/19

This round only locates text and code; nothing is re-proven.

- Pointwise objective/minimum distinction, beta=0 exception: corresponding content exists in the 9/19 revision decisions and source.
- Exact analytic small cluster, unified remainder, nonzero direction M0: corresponding content exists in the 9/19 theory report and main manuscript.
- SSO rank-one normal center agreement: the 9/19 main manuscript explicitly states K_F=K_R, including C=0 and endpoints. 9/13 once reported a stronger independent corollary that "all admissible profiles agree"; this round does not treat the two statements as automatically equivalent, and it needs to be confirmed later whether an explicit proof should be added.
- s_delta/delta² → 3/16 for the non-transversal 3×3 example: kept in this package's context; the example was not located in the 9/19 main manuscript sections examined, so it is not claimed to be in the current manuscript.
- 69-item generator: currently uses 9/19's own 14 items and publication_numbers structure; the two sets of scripts are not the same entry point.
- pseudo-Huber training reference solver: not implemented in the 9/19 GPU directory; finite NS must not be used as a stand-in for it.
- Official template: on 9/13 the download returned 403 and was not compiled; 9/19 has a SIAM-class PDF and an explicit statement of third-party academic source dependencies. The status has changed, so one can no longer say categorically "SIAM class was never compiled", nor claim that the complete official package has been verified.

## 4. Three tiers of evidence labels

1. **Recovered/checked this time**: archive bytes, CRCs, manifests, and file hashes obtained; key theory and validation reports and the actual GPU implementation read; the standalone GPU and the GPU inside the revision compared and found byte-for-byte identical.
2. **Historical executions within the archive**: 9/19's 14 math checks, CPU 18-case E0, five-method training/resume, real TinyStories micro pipeline. They have historical records but were not re-run this round.
3. **Conversation records only**: 9/13's 23/11 pages, 69 results, seven groups of direction checks, 3 CPU steps, and claims about the existence of the corresponding source. The originals are missing and cannot be upgraded to evidence of this round.

None of the three tiers amounts to a complete formal proof, a comprehensive originality certification, or a guarantee of journal acceptance.
