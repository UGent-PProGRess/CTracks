# CTracks

CTracks tracks tracer particles through dynamic micro-CT scans to recover fluid flow velocity fields.


* `src/ctrex`: the core CT reconstruction/simulation framework — handles the static CT side (projectors, reconstruction, sample description).
* `src/ctracks`: builds on `ctrex` to add tracer particles as a dynamic sample feature — projecting, tracking, and post-processing their motion through a scan.
* `scripts/particle_tracking`: reconstruction and analysis scripts that drive `ctracks`/`ctrex` end-to-end — simulating tracer motion, reconstructing scans, and analyzing/visualizing the resulting tracks.

Archived on Zenodo: [![DOI](https://zenodo.org/badge/1366402682.svg)](https://doi.org/10.5281/zenodo.22712231)

# Usage
=======

After installation (see below), create a python script in the folder scripts/ and run it.
The most convenient way is to copy an existing file that does what you need to do and adjust where necessary.
If you're new to the codebase and want to reconstruct particle trajectories from your own porous-sample
scan, start from `scripts/particle_tracking/recon_demo.py` - a single-dataset, heavily-commented version
of the particle-tracking reconstruction, with every setting you're likely to need explained inline.
CTrex is built as modular as possible for academic purposes, so everything can be tweaked.
As a result, the scripts can combine submodules in every possible way.
To simplify the scripts, it is strongly advised to define common templates in each subproject.
This is evident when scripts have a lot of repeated code compared to other scripts.

In general, the goal is to create a representation of the (dynamic) sample for a given set of CT projections.
In iterative reconstruction, this is done by initializing the representing parameters (e.g. attenuation coefficients),
and optimizing them so a forward CT simulation resembles the given projections as closely as possible.
CTrex uses (custom) differentiable PyTorch modules, which streamlines the updates of those parameters massively.
For a loss L(output = projection(input); measured), each operator computes the gradients of the input
to backpropagate the gradients of the output. This tells us, if we want to minimize this loss
(= update L with a decreasing step = negative gradient), how much we should update the input. 

# Installation

New to Python, virtual environments, or CUDA? Don't worry - every step below links to the
official installer/instructions for that piece, so you don't need prior experience with it.
There are two installation paths:
* **Basic (CPU)**: simplest to set up, works on any machine. Ray tracing with the custom
  CUDA kernels isn't available; ray integrals fall back to PyTorch samplers. Particle
  tracking works fine on CPU, just slower than on a GPU.
* **GPU (CUDA)**: needed for the custom CUDA ray-tracing kernels and for reconstruction
  speeds practical on real datasets. Requires an NVIDIA GPU and a few extra driver/toolkit
  installs (all linked below).

If you're unsure which one you need: start with the Basic installation, [verify it works](#verifying-your-installation),
and switch to GPU later if reconstructions are too slow - the only difference is which
`torch`/`cupy` builds you install.

### 0. Prerequisites

* **Python 3.9 or newer.** If you don't already have it, install it from
  [python.org](https://www.python.org/downloads/) (on Windows, tick "Add python.exe to PATH"
  during setup).
* **A virtual environment** keeps this project's packages separate from everything else on
  your machine - strongly recommended. From the project's root folder:
  ```
  python -m venv .venv
  ```
  Then activate it (do this every time you open a new terminal to work on this project):
  * Windows (cmd/PowerShell): `.venv\Scripts\activate`
  * macOS/Linux: `source .venv/bin/activate`

  Your terminal prompt should now start with `(.venv)`.

### 1. Basic (CPU) installation

```
pip install -e .[cpu]
```
This installs CTrex itself (editable, so it tracks the source code directly) plus PyTorch
and `cupy`. Note: `cupy` is required even for this CPU-only path (one internal module always
imports it) - on most systems `pip` fetches a working build automatically, but if it fails,
installing the [CUDA toolkit](#2-gpu-cuda-installation-optional) below (step 2) will resolve it even
if you don't intend to use the GPU.

### 2. GPU (CUDA) installation (optional)

1. **NVIDIA driver**: install the latest driver for your GPU from
   [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) (choose a "Studio" driver
   over "Game Ready" if offered - it's more stable for compute workloads).
2. **Check your GPU's CUDA compute capability** against the
   [CUDA Wikipedia page](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) - you'll need this
   to pick a compatible CUDA/PyTorch version below. GPUs older than compute capability 7.5
   are not supported by CUDA 13.
3. **(Windows only) Visual Studio Build Tools**: install from
   [visualstudio.microsoft.com](https://visualstudio.microsoft.com/downloads/) (the free
   "Build Tools for Visual Studio" is enough) and select the "Desktop development with C++"
   workload. This provides the MSVC compiler needed to build the CUDA kernels.
4. **CUDA toolkit**: check [PyTorch's supported versions](https://download.pytorch.org/whl/torch/)
   first (search that page for `+cu1` to see which CUDA versions have prebuilt PyTorch
   wheels) - then download the matching version from the
   [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). Avoid picking
   the newest CUDA release unless you've confirmed PyTorch already supports it.
5. **Install the Python packages** (with your virtual environment still active):
   ```
   pip install -e .
   pip install -r requirements.txt
   ```
   Then install PyTorch and `cupy` built for your specific CUDA version - use
   [PyTorch's official installer selector](https://pytorch.org/get-started/locally/) to get
   the exact command for your CUDA version (it will look like the example below, but check
   the site rather than copying this verbatim, since versions change):
   ```
   python -m pip install torch==2.7.1+cu128 --no-cache --index-url https://download.pytorch.org/whl/cu128
   python -m pip install cupy-cuda12x
   ```
   (`cupy-cuda12x` covers any CUDA 12.x toolkit; use `cupy-cuda11x` etc. if you installed a
   different major version. If `cupy` still fails to import, see its
   [installation troubleshooting guide](https://docs.cupy.dev/en/stable/install.html).)

### 3. Particle-tracking extras (optional)

If you'll use the `scripts/particle_tracking` analysis/visualization scripts (classical
particle linking, 3D plots), install their extra dependencies too:
```
pip install -e .[particle-tracking]
```

### Verifying your installation

Run the installation smoke test - it fabricates a small set of known particles, reconstructs
them, and checks that most are recovered correctly:
```
pip install -e .[dev]
pytest tests/test_ctracks/test_installation.py -v
```
A pass means your Python/PyTorch/CUDA setup can run the reconstruction pipeline end to end.
If it fails, the error is almost always an installation problem (see the printed message) -
recheck the steps above, in particular that `torch.cuda.is_available()` returns `True` if you
installed the GPU path (run `python -c "import torch; print(torch.cuda.is_available())"`).

For a more thorough (and much slower) check - a full reconstruction of a synthetic
capillary-flow dataset, using every reconstruction callback, checked against the known
Poiseuille flow profile - run:
```
pytest tests/test_ctracks/test_capillary_integration.py -v -m slow
```
It also saves a plot comparing the reconstructed and theoretical velocity profiles to
`tests/test_ctracks/output/`. This one is a deeper confidence check, not a routine
install check - the installation test above is enough for everyday use.

