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

```