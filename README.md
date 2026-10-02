# 4Denoise

4Denoise is a research-oriented Python toolkit for loading, visualizing,
denoising, and analyzing four-dimensional scanning transmission electron
microscopy (4D-STEM) data. Its main `HyperData` object represents a scan as
`(Ry, Rx, Ky, Kx)`: two scan axes followed by two diffraction axes.

The included mini dataset and DEMO notebook are the fastest way to try the
package. The full 3D strain-mapping study uses separate, much larger data and
optional simulation and MATLAB tools.

## Install

The supplied Conda environment uses Python 3.11. From a terminal:

```bash
git clone https://github.com/mirelesadan/4Denoise.git
cd 4Denoise
conda env create -f environment.yml
conda activate 4denoise-main
python -m pip install -e .
python -c "from fourdenoise import HyperData; print(HyperData.__name__)"
```

For a pip-only setup, use `python -m pip install -e ".[notebook]"` in a
Python 3.10+ environment. The study's simulation workflow additionally needs
`python -m pip install -e ".[simulation]"`; BM3D/BM4D support is optional via
`.[bm]`. An editable install uses the current local source when you edit it.

## Quick Example

Run this in Jupyter from the repository root, where
`mini_dataset_binned.npy` is included:

```python
from fourdenoise import HyperData

data = HyperData("mini_dataset_binned.npy")
print(data.shape)  # (15, 13, 128, 128)

image = data.virtual_image(
    annulus=(37, 50), detector_units="pixels", show=False,
)
image.show(title="Annular virtual image")
```

For a fuller verified example, run
[the DEMO notebook](DEMO_exp_4dstem_ripple_processing.ipynb) from a fresh
Jupyter kernel with `jupyter lab`.

## Explore

- [Documentation website](https://mirelesadan.github.io/4Denoise/): installation, task guides, a curated API reference, and the case study.
- [3D strain-mapping case study](https://mirelesadan.github.io/4Denoise/case-studies/3d-strain-mapping.html): external data, experimental notebook, simulation, and MATLAB GUI.
- [Experimental processing notebook](exp_MoS2_MoSe2_processing.ipynb): the full, compute-intensive analysis.
- [Simulation notebook](generateSimulatedRipple_4Ddata.ipynb): the companion ripple simulation workflow.

The large experimental and simulated datasets are not bundled with this
repository; they are available from the
[Zenodo data record](https://zenodo.org/records/17246822). The case study
explains which stages have been verified. The documentation guide can be built
locally with `python -m pip install -e ".[docs]"` followed by
`python -m sphinx -b html -W --keep-going docs docs/_build/html`. Its
[source pages](docs/index.md) are maintained in this repository. GitHub Pages
publishes the validated `main` branch automatically; PyPI publication is not
required for either the website or the editable install above.

To run the local unit tests: `python -m unittest discover -s tests`.

Pull requests also check the documentation build, guide links, and a
fresh-kernel run of the mini-data DEMO. See the
[documentation maintenance notes](docs/README.md#automated-checks-and-maintenance)
for local commands and downloadable CI reports. The large experimental
workflow is not executed by these checks.
