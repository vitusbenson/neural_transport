# NeuralTransport

*A Python library to train neural network emulators of atmospheric transport models.*

<a href='https://pypi.python.org/pypi/neural-transport'>
    <img src='https://img.shields.io/pypi/v/neural-transport.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://twitter.com/vitusbenson" target="_blank">
    <img src="https://img.shields.io/twitter/follow/vitusbenson?style=social" alt="Twitter">
</a>




# Installation


```
conda create -n neuraltransport python=3.10
conda activate neuraltransport
conda install -c conda-forge ffmpeg pkg-config libjpeg-turbo opencv cupy cuda-version=11.8
conda install -c conda-forge numpy pandas xesmf cdo python-cdo xarray dask zarr netCDF4 bottleneck matplotlib seaborn cartopy shapely xskillscore xrft pyarrow
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip3 install lightning cdsapi pypdf2 trimesh rtree ipykernel ipywidgets tensorboard einops timm ecmwf-api-client eccodes dm-tree cfgrib 
pip3 install git+https://github.com/jbusecke/xmovie.git
pip3 install git+https://github.com/vitusbenson/torch_advection.git
pip install -e .
```


# Cite NeuralTransport

In case you use NeuralTransport in your research or work, it would be highly appreciated if you include a reference to our [paper](https://link.to.paper) in any kind of publication.

```bibtex
@article{benson2024neuraltransport,
  title = {Atmospheric transport modeling with neural networks},
  author = {Vitus Benson et al},
  journal = {TBD},
  publisher = {TBD},
  year = {2024},
  volume = {1},
  number = {1},
  pages = {1},
  doi = {XXX.YYY/xxx.yyy},
  url = {https://doi.org/XXX.YYY/xxx.yyy},
}
```

# Contact

For questions or comments regarding the usage of this repository, please use the [discussion section](https://github.com/vitusbenson/neural_transport/discussions) on Github. For bug reports and feature requests, please open an [issue](https://github.com/vitusbenson/neural_transport/issues) on GitHub.
In special cases, you can reach out to Vitus (find his email on his [website](https://vitusbenson.github.io/)).