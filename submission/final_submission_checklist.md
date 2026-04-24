# Final Submission Checklist

## Current Verified State

- Main PDF: `manuscript/main.pdf`, 39 pages.
- Supplementary PDF: `manuscript/supplementary.pdf`, 26 pages.
- Main text references: 40 cited references, all resolved in the current build.
- Supplementary references: 6 cited references, all resolved in the current build.
- Current build has no unresolved citation or BibTeX warnings. Remaining LaTeX issues are layout warnings only.

## Recommended Order

1. Do one final scientific read of the PDF, not the TeX.
2. Create the confidential review-only data/code repository link.
3. Submit to Nature Physics through the submission portal.
4. After the journal submission confirmation arrives, submit the arXiv preprint.
5. When the arXiv ID posts, update the Nature record if the portal allows it.

## Submit To Nature Physics

Initial submission package:

- `manuscript/main.pdf`
- `manuscript/supplementary.pdf`
- Cover letter text from `submission/nature_physics_cover_letter.md`
- Review-only data/code link, with clear access instructions.
- Optional suggested/excluded reviewers using `submission/editorial_portal_text.md`

Do not submit private credentials, `dwave.conf`, `.env`, QPU account information, cache folders, Python bytecode, or nonessential operational logs.

## Final Scientific Review

Check these claims carefully in the PDF:

- The manuscript consistently says reverse-anneal open-system relaxation, not isolated-system ETH.
- It does not imply quantum advantage.
- The conditional Boltzmann marginal claim is limited to relaxed regimes and calibrated effective temperatures.
- The Glauber comparison is described as a matched-temperature single-spin-flip baseline, not a proof against all classical dynamics.
- The disorder and atypical-environment results are framed as memory arrest under tested conditions, not as a universal MBL transition.
- The environment-size threshold is described as geometry and coupling dependent, not a universal qubit count.
- All D-Wave solver names, dates, beta values and auto-scale caveats are consistent between Results, Methods and Supplementary Information.

## Final Compliance Review

Confirm:

- Data availability and Code availability statements match the actual review-only repository.
- LLM disclosure is acceptable and accurate.
- Funding, competing interests and correspondence fields are correct.
- The author affiliation is exactly how you want it to appear.
- The manuscript includes no `TODO`, `FIXME`, placeholder URLs, private file paths, credentials, or informal notes.
- Figure labels and captions match the figures in the PDF.
- Every figure has source data or enough raw/processed data in the review package to regenerate it.

## Plagiarism And Overlap

I did not find an obvious duplicate paper in the literature search, and the current reference graph is internally consistent. That is not the same as a formal plagiarism/similarity certification. Before submission, run one formal similarity check if you have access to iThenticate/Crossref Similarity Check or an institutional equivalent.

Manual overlap check:

- Re-read the Introduction and Methods for phrases copied too closely from D-Wave documentation or review articles.
- Keep standard technical phrases where unavoidable, but rewrite any long borrowed sentence structures.
- Ensure all close context papers are cited, especially work on thermalization/ETH, D-Wave effective temperature and Gibbs sampling, reverse annealing, memory erasure, programmable localization and Rydberg bath-size ergodicity.

## Commands For Final Local Verification

Run from `nature-papers/locth1/manuscript`:

```sh
latexmk -pdf -interaction=nonstopmode main.tex supplementary.tex
```

Then run from `nature-papers/locth1`:

```sh
rg -n "undefined|Citation .* undefined|Warning--" manuscript/*.log manuscript/*.blg
rg -n "TODO|FIXME|PLACEHOLDER|INSERT|private|credential|dwave.conf|\\.env" manuscript submission
```

If the second command only finds intentional bracketed placeholders in `submission/`, replace those before copying text into the portal.

## If Nature Physics Rejects Without Review

Use the decision letter to decide the next venue. My recommended order is:

1. PRX Quantum if you want a physics/quantum-information audience and can tolerate another selective review path.
2. Nature Communications if the editors offer transfer or if you want broad visibility with a slightly less narrow editorial threshold.
3. Communications Physics if you want the most direct Springer Nature transfer route and a physics-focused audience.

Do not rewrite the paper immediately after a desk rejection unless the editor identifies a specific framing problem. The fastest useful action is usually to adapt the cover letter and submit the same technically checked package to the next venue.
