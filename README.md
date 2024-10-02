Support FlexAttention in FlashAttention3

Build environment to benchmark FlashAttention3
```bash
mkdir -p envs
mamba create -p ~/envs/flashattn_3 python=3.10 -y
ln -s ~/envs/flashattn_3 ./envs/flashattn_3
mamba activate ./envs/flashattn_3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu123
MAX_JOBS=80 pip install flash-attn --no-build-isolation
```

Build environment to benchmark FlexAttention in nightly build torch
```bash
mkdir -p envs
mamba create -p ~/envs/flexattn python=3.10 -y
ln -s ~/envs/flexattn ./envs/flexattn
mamba activate ./envs/flexattn
mamba install cmake ninja rust -y

# Install nightly build torch
git clone https://github.com/pytorch/pytorch.git
cd pytorch && \
    git submodule sync && \
    git submodule update --init --recursive && \
    cd -

cd pytorch && \
    MAX_JOBS=80 _GLIBCXX_USE_CXX11_ABI=1 CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which mamba))/../"} python setup.py develop && \
    cd -

```

Build environment to benchmark FlexAttention in nightly build torch
```
mamba create -p ~/envs/flexattn-nightly-built python=3.10 -y
mamba activate ~/envs/flexattn-nightly-built
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia -y
mamba install ipython nvitop -y

```