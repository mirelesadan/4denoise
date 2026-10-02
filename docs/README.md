# 4Denoise documentation plan

This is the editorial map for the documentation work, not a user-facing guide
page. The guide entry is `docs/index.md`; the Sphinx/MyST website builds
locally but has not been published. This plan does not appear in site navigation.

## Purpose and audience

The guide should let a new user install 4Denoise, understand the data model,
run a small example, complete a task, and find the relevant API contract.
Experienced users should be able to reproduce the MoS2-MoSe2 analysis without
mistaking its material-specific choices for package defaults.

The documentation serves three audiences:

- New 4D-STEM users who need a short, runnable path from data loading to a plot.
- Analysts who need task-oriented recipes and precise units, shapes, metadata,
  memory, and return-value behavior.
- Researchers reproducing the 3D strain-mapping study with its external data
  and simulation requirements.

## Support boundary

The core guide covers `HyperData`, `RealSpace`, and `ReciprocalSpace` for
loading/saving, calibration, preprocessing, visualization, denoising,
unfolding, peak analysis, and strain results. Explain which operations accept
2D images, 3D stacks, or 4D scans instead of implying they all accept every
shape. The 3D reconstruction workflow, abTEM-based simulation, and MATLAB GUI
are optional or study-specific material, not requirements for basic use.
Private implementation classes and unfinished helpers are not public API.

## First-release navigation

| Section | Proposed page | What it answers |
| --- | --- | --- |
| Home | `docs/index.md` | What 4Denoise does; where to begin; links to the guide, API, and study. |
| Getting started | `docs/getting-started/install.md` | Supported Python environment, core install, optional extras, and import check. |
| Getting started | `docs/getting-started/quickstart.md` | A short, clean-kernel workflow using the included `mini_dataset_binned.npy`. |
| Concepts | `docs/concepts/data-model.md` | `(Ry, Rx, Ky, Kx)`, 2D/3D/4D distinctions, units, metadata, and mutation/copy behavior. |
| Guides | `docs/guides/io-and-calibration.md` | Supported file formats, HDF5 selection/chunks, saving, and coordinate calibration. |
| Guides | `docs/guides/visualization.md` | Diffraction patterns, virtual images, real-space views, masks, and polar views. |
| Guides | `docs/guides/denoising-and-unfolding.md` | Choosing a denoiser, 4D-to-3D traversal, reversible folding, and rank diagnostics. |
| Guides | `docs/guides/peaks-and-strain.md` | Peak detection/refinement, intensity integration, strain inputs, and interpretation. |
| API reference | `docs/reference/hyperdata.rst` | Public `HyperData` methods and their exact signatures. |
| API reference | `docs/reference/views-and-results.rst` | `RealSpace`, `ReciprocalSpace`, `PeakDetectionResult`, and `StrainResult`. |
| API reference | `docs/reference/utilities.rst` | Selected public standalone functions, including `read_4D` and `plot_traversals`. |
| Case study | `docs/case-studies/3d-strain-mapping.md` | Data citation, assumptions, processing notebook, reconstruction, and MATLAB GUI. |
| Help | `docs/troubleshooting.md` | Installation, environment selection, file loading, calibration, and memory issues. |

Start with these pages rather than documenting every private function. Split a
guide later only when its scope becomes too large. Optional simulation helpers
can have their own reference page after their supported API is reviewed.

## What belongs where

- `README.md` is the concise repository landing page: purpose, install, one
  working example, dataset credit, and links to documentation and notebooks.
  Detailed study and MATLAB material now lives in the case study.
- The DEMO notebook is the runnable teaching example. It uses the repository's
  mini dataset and relative paths, imports explicit names, and runs top to
  bottom in a fresh kernel. The obsolete peak call was replaced by a verified
  `get_peaks(...)` and center/intensity/strain workflow.
- The experimental notebook is a reproducibility case study, not the
  installation test. A locally modernized version was smoke-tested through
  the initial crop, but its edits are not included in this documentation PR.
  The case-study page distinguishes that local verification from the published
  historical notebook, which still needs migration and full validation.
- The simulation notebook and MATLAB GUI remain linked from the case study.
  They are not prerequisites for the core quickstart.
- API pages describe supported public classes and functions. Private helpers
  such as `_DenoisingMethods` and `_DenoiseEngine` remain implementation
  details. Signatures and docstrings are the source for API facts; guides
  explain when and why to call them.

## Documentation rules

- Every executable example declares its input shape, units, package extras,
  and whether it is quick or compute-intensive.
- Use current method names and verify return types against the code. Do not
  copy historical notebook cells into guides without running them.
- Explain when an operation returns a new object, retains metadata, changes
  calibration, or keeps cropped-out data only in unfolding metadata.
- Keep the distinction between an included example, an external dataset, and
  a synthetic illustration visible to the reader.
- Preserve the study's attribution and data citation when moving content.
  Avoid embedding multi-gigabyte data or large executed outputs in the site.

## Verification and release gates

1. Run the DEMO from a clean kernel in the documented environment; use its
   verified lightweight path for the getting-started tutorial.
2. Run the experimental notebook against the actual external dataset on a
   suitable machine. Record package version, input data, and any long-running
   or optional steps; do not make this a routine CI job.
3. Build the Sphinx/MyST site locally with the package installed. Start
   autodoc with the core module; optional simulation imports need their own
   dependency handling.
4. Add CI checks for the documentation build, unit tests, and a bounded
   example. Publish the verified site through GitHub Pages afterward.

Publishing to PyPI is not a prerequisite: the repository already has a
`pyproject.toml` and supports editable installation.

## Automated checks and maintenance

Run these from the repository root in an isolated Python 3.11 environment:

```bash
python -m pip install -e ".[docs,docs-test]"
python -m pip check
python -m unittest discover -s tests
python -m sphinx -b html -W --keep-going docs docs/_build/html
python -m sphinx -b linkcheck -W --keep-going docs docs/_build/linkcheck
python scripts/execute_demo.py
```

The documentation workflow runs on every pull request and push to `main`.
`docs-build` treats Sphinx warnings as errors; `docs-links` checks external
destinations with bounded retries; unavailable HTTP links must not silently
pass as ignored. The HTML build checks internal document
references. Fix invalid destinations rather than broadly ignoring failures;
external sites can also fail temporarily, in which case inspect the report
before retrying the check.

`demo-notebook` executes only `DEMO_exp_4dstem_ripple_processing.ipynb` and the
included full mini dataset. It starts a fresh kernel using the runner's Python
interpreter, discards cached outputs, and rejects cell errors or skipped code.
Cells have a 120-second timeout (`--timeout` can override it locally), and each
CI job has a ten-minute limit. The tracked notebook is never overwritten.

Download the HTML preview, link report, and executed notebook from the
workflow's artifacts (retained seven days). On notebook failure, the report
contains partial output when execution reached the kernel. Local reports go
to ignored `docs/_build/`. Research notebooks and external experimental data
are deliberately excluded; successful CI is not full scientific validation
of the 3D reconstruction workflow.

Update the guide and lightweight example alongside API changes. Publishing
the verified site through GitHub Pages is a separate next step; this workflow
does not deploy a website.
