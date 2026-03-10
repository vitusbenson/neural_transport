![#](https://raw.githubusercontent.com/vitusbenson/neural_transport/main/logo.png)

*A Python library to train neural network emulators of atmospheric transport models, extended with a data assimilation and sampling framework based on Flow Matching.*

<a href='https://pypi.python.org/pypi/neural-transport'>
    <img src='https://img.shields.io/pypi/v/neural-transport.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://twitter.com/vitusbenson" target="_blank">
    <img src="https://img.shields.io/twitter/follow/vitusbenson?style=social" alt="Twitter">
</a>

<a href="https://arxiv.org/abs/2408.11032" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2408.11032-b31b1b.svg" alt="ArXiv">
</a>



# Installation

```
conda create -n neuraltransport -c conda-forge python=3.12
conda activate neuraltransport
conda install -c conda-forge ffmpeg pkg-config libjpeg-turbo opencv cupy
conda install -c conda-forge numpy pandas xesmf cdo python-cdo xarray dask zarr netCDF4 bottleneck matplotlib seaborn cartopy shapely xskillscore xrft pyarrow
pip3 install torch torchvision torchaudio
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
pip3 install lightning cdsapi pypdf2 trimesh rtree ipykernel ipywidgets tensorboard einops timm ecmwf-api-client eccodes dm-tree cfgrib pynvml wandb ruamel.yaml moviepy torch_harmonics tensorly tensorly-torch
pip3 install git+https://github.com/jbusecke/xmovie.git
# Go inside your neural_transport folder (cd neural_transport)
pip install -e .

# Development tools (linting, testing, pre-commit hooks)
pip install ruff pytest pytest-cov pre-commit
pre-commit install
```


# Cite NeuralTransport

In case you use NeuralTransport in your research or work, it would be highly appreciated if you include a reference to our [paper](https://arxiv.org/abs/2408.11032) in any kind of publication.

```bibtex
@article{benson2024neuraltransport,
  title = {Atmospheric Transport Modeling of CO2 with Neural Networks},
  author = {Vitus Benson, Ana Bastos, Christian Reimers, Alexander J. Winkler,
 Fanny Yang and Markus Reichstein},
  eprint={2408.11032},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2408.11032},
}
```

# Contact

For questions or comments regarding the usage of this repository, please use the [discussion section](https://github.com/vitusbenson/neural_transport/discussions) on Github. For bug reports and feature requests, please open an [issue](https://github.com/vitusbenson/neural_transport/issues) on GitHub.
In special cases, you can reach out to Vitus (find his email on his [website](https://vitusbenson.github.io/)).
