### Set up the python environment

```
conda create -n snake python=3.7
conda activate snake

# make sure that the pytorch cuda is consistent with the system cuda
# e.g., if your system cuda is 9.0, install torch 1.1 built from cuda 9.0
pip install torch==1.1.0 -f https://download.pytorch.org/whl/cu90/stable

pip install Cython==0.28.2
pip install -r requirements.txt

# install apex
cd
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 39e153a3159724432257a8fc118807b359f4d1c8
export CUDA_HOME="/usr/local/cuda-9.0"
python setup.py install --cuda_ext --cpp_ext
```

### Compile cuda extensions under `lib/csrc`

```
ROOT=/path/to/snake
cd $ROOT/lib/csrc
export CUDA_HOME="/usr/local/cuda-9.0"
cd dcn_v2
python setup.py build_ext --inplace
cd ../extreme_utils
python setup.py build_ext --inplace
cd ../roi_align_layer
python setup.py build_ext --inplace
```


