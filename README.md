![Test Status](https://img.shields.io/badge/Tests-Passed-brightgreen)
![Conda Environment](https://img.shields.io/badge/environment-conda-green?logo=anaconda)


# GWBird 

GWBird is a Python package that allows computing the response and sensitivity of a network of detectors to a stochastic gravitational wave background.



<p align='center'>
   <img src='logo.png' alt='logo' width='210'>
</p>

## Installation

### Requirements

To use GWBird, make sure you have the following Python packages installed:

- `numpy`
- `healpy`
- `scipy`
- `mpmath`
- `matplotlib`

1. Create a new conda environment with Python ≥ 3.10
```sh
conda create -n myenv python=3.11
```

2. Activate the environment
```sh
conda activate myenv
```
3. Install scientific packages
```sh
conda install numpy scipy mpmath matplotlib ipykernel
conda install -c conda-forge healpy
```
4. Install PINT (for pulsar timing analysis) from GitHub
```sh
pip install git+https://github.com/nanograv/PINT.git
```
5. (Optional) Register the environment as a Jupyter kernel
```sh
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
```
6. Verify installation
```sh
python -c "import numpy, healpy, scipy, mpmath, matplotlib, pint; print('✅ All packages installed correctly.')"
```

### Download and Installation

1. Clone or download this repository to your computer:

   ```sh
   git clone /https://github.com/ilariacaporali/GWBird
   cd GWBird
   ```

2. Install the package in your selected environment by running:

   ```sh
   pip install .
   ```

3. Now you can import GWBird in your Python scripts:

   ```python
   import gwbird
   ```

## Credits

If you use this code in your research, we kindly ask you to cite the paper: **[insert reference]**.

---

For more information, visit the [official documentation](#), open an issue in the repository or contact via mail at ilaria.caporali@phd.unipi.it.


