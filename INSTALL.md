# Installation

This project depends on the following main libraries: `pytorch>=1.3`, `faiss-gpu>=1.4` and `MinkowskiEngine>=0.4`. You will need a full installation of CUDA 10.1.243 in order to compile MinkowskiEngine (note that the installation can be performed on any directory by specifying a custom path. Your chosen path needs to be specified with the CUDA_HOME environment variable). 

One possible way to install these three libraries (and some other relevant ones) on Linux-64 through Anaconda and Pip is by using the following commands:

```
# create new environemnt and install compiler and other tools
conda create -n sparsencnet python=3.6.9=h265db76_0
conda activate sparsencnet
conda install gcc-5 -c psi4
conda install numpy openblas
conda install libstdcxx-ng -c anaconda

# set environment variables for the compilation of MinkowskiEngine
export CUDA_HOME=/your_path_to/cuda-10.1.243
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64":"${CONDA_PREFIX}/lib":"/usr/lib/x86_64-linux-gnu/"
export PATH="${CONDA_PREFIX}/bin":"${CUDA_HOME}/bin":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CPP="${CONDA_PREFIX}/bin/g++ -E"
export CXX="${CONDA_PREFIX}/bin/g++"
export LIBRARY_PATH=$LD_LIBRARY_PATH
export PYTHONPATH="${CONDA_PREFIX}/lib/python3.6/site-packages/"

# install PyTorch and ME
pip install torch torchvision
pip install -U MinkowskiEngine # compilation may take a while

# download cuda8 runtime libs which are required by faiss-gpu. As the dependencies for the faiss-gpu package are broken we do this manually.
conda install https://anaconda.org/anaconda/cudatoolkit/8.0/download/linux-64/cudatoolkit-8.0-3.tar.bz2
conda install --force --no-deps faiss-gpu=1.4.0=py36_cuda8.0.61_1 -c pytorch

# install some additional libraries
conda install matplotlib scikit-image pandas

# replace pillow with pillow-simd
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# install jupyter lab for evaluation on HPatches-Seq
conda install -c conda-forge jupyterlab

```

With this newly created environment, you can now clone this repo and start using it.
