

Support FlexAttention in FlashAttention3


## Build environment

These instructions currently only tested in CUDA 12.4. We eventually need CUDA 12.3 for the build, and this is still under dev.


Install mamba
```bash
# https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Build environment to benchmark FlashAttention3 with nightly torch
```bash
mkdir -p envs
mamba create -p ~/envs/flashattn_3 python=3.10 -y
ln -s ~/envs/flashattn_3 ./envs/flashattn_3
mamba activate ./envs/flashattn_3
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia -y
mamba install ipython nvitop ninja cmake -y
pip install packaging
# Install FlashAttention3
MAX_JOBS=80 python setup.py install
cd hopper
MAX_JOBS=80 python setup.py install
```


Build environment to benchmark FlexAttention in nightly build torch
```bash
mkdir -p envs
mamba create -p ~/envs/flexattn-nightly-built python=3.10 -y
mamba activate ~/envs/flexattn-nightly-built
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia -y
mamba install ipython nvitop -y
```

Build environment for Flex attention gym
```bash
mamba activate ~/envs/flexattn-nightly-built
git clone https://github.com/pytorch-labs/attention-gym.git
cd attention-gym
pip install .
```