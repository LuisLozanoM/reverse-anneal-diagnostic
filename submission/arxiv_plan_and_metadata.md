# arXiv Plan And Metadata

## Recommendation

Submit to Nature Physics first, then post to arXiv after you have the journal submission confirmation, ideally the same day or the next morning. Nature Portfolio allows the original submitted version to be posted on preprint servers, so arXiv does not block consideration. The practical reason to submit Nature first is version control: once arXiv announces the paper, fixes require a new public version.

Use arXiv before journal submission only if priority is urgent and the files are already frozen.

## Category

Primary: `quant-ph`

Cross-list: `cond-mat.stat-mech`

Optional cross-list, if the form permits and it feels natural: `cond-mat.dis-nn`

Rationale: the hardware and protocol are quantum annealing/quantum simulation, while the scientific question is subsystem relaxation, thermal marginals, disorder and memory arrest.

## arXiv Metadata

Title:
Subsystem relaxation in a programmable quantum annealer

Author:
Luis Lozano

Comments:
39 pages, 6 figures; 26-page Supplementary Information.

Journal reference:
Leave blank at first submission.

DOI:
Leave blank at first submission.

License:
Use either the arXiv non-exclusive distribution license or CC BY 4.0 if you want maximum reuse. If you want the simplest conservative choice before journal acceptance, use the arXiv non-exclusive distribution license.

## Abstract

Small subsystems are expected to relax when coupled to sufficiently large environments, but direct large-scale tests with independent control over bath size, coupling and disorder remain scarce. Here we realise a controllable subsystem-environment experiment on two programmable quantum annealers by coupling a six-qubit subsystem to an on-chip environment during reverse annealing. We find that the subsystem becomes initial-state independent within experimental resolution as the environment grows or the coupling increases, whereas quenched disorder and atypical environment preparation arrest this relaxation. In the relaxed regime, the final subsystem readout is accurately described by a calibrated conditional Boltzmann marginal across graph geometries, coupling strengths, disorder levels and two QPU generations. In a mixed-frustration regime, matched-temperature single-spin-flip Glauber dynamics fails to reproduce the observed relaxation rate, consistent with a transverse-field-assisted relaxation pathway. These results separate the dynamics of relaxation from the form of the relaxed readout and establish programmable quantum annealers as controllable platforms for studying subsystem relaxation, effective thermal readout, and memory arrest.

## Source Package Contents

For arXiv, prefer TeX source rather than uploading the compiled PDF. A minimal source package should contain:

- `main.tex`
- `phase6_results.tex`
- `main.bbl`
- `nature.cls`
- `figures/fig1_combined.pdf`
- `figures/fig3_subsystem.pdf`
- `figures/fig4_disorder.pdf`
- `figures/fig5_baselines.pdf`
- `figures/fig6_thermal_marginal.pdf`
- `figures/fig7_barrier_crossing.pdf`

For Supplementary Information, the simplest path is to upload `supplementary.pdf` as a supplementary/ancillary file if arXiv accepts it in the workflow. If you want the SI compiled by arXiv too, package:

- `supplementary.tex`
- `supplementary.bbl`
- all supplementary figures referenced by `supplementary.tex`

Do not include `.aux`, `.log`, `.blg`, `.fdb_latexmk`, `.fls`, `.out`, `.toc`, local credentials, `dwave.conf`, `.env`, cache folders, or any private operational logs.

## arXiv Technical Checks

Before uploading, build the exact arXiv bundle locally from a clean directory. Check that:

- `pdflatex main.tex` finds `phase6_results.tex`, `nature.cls`, `main.bbl` and all figures.
- The compiled PDF has the same title, authors, abstract, figures and references as `manuscript/main.pdf`.
- No figure path points to an absolute local path.
- File names use only safe characters: letters, numbers, underscore, hyphen, period.
- The arXiv preview PDF is inspected page by page before final submission.

## After arXiv Posts

Update the Nature submission record, if the portal allows it, with:

A preprint version of this manuscript is now available at arXiv:[INSERT ARXIV ID]. This is the author-submitted version and the manuscript remains under consideration at Nature Physics.
