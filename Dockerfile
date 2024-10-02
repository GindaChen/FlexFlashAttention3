
FROM mambaorg/micromamba:1.5.10-jammy-cuda-12.3.2




# docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
# # Build torch nightly from source to support CUDA 12.3
# FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# RUN mkdir -p /app/bin
# WORKDIR /app

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     git \
#     wget \
#     curl \
#     ca-certificates \
#     htop \ 
#     tmux \ 
#     tree \
#     vim \
#     sudo

# # Download Micromamba
# RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2
# RUN tar -xvjf /tmp/micromamba.tar.bz2 -C /app
# RUN ln -s /app/bin/micromamba /usr/local/bin/micromamba && \
#     ln -s /app/bin/micromamba /usr/local/bin/mamba

# RUN mamba create -n pytorch python=3.10 

# SHELL ["/bin/bash", "-c"]

# RUN micromamba shell init --shell bash --root-prefix=/root/micromamba \
#     && mamba activate pytorch \
#     && mamba install cmake ninja rust

# RUN git clone https://github.com/pytorch/pytorch.git

# RUN cd pytorch && \
#     git submodule sync && \
#     git submodule update --init --recursive && \
#     cd -
    
# RUN mamba activate pytorch && cd pytorch && _GLIBCXX_USE_CXX11_ABI=1 CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which mamba))/../"} python setup.py develop


# # RUN micromamba shell init --shell bash --root-prefix=/root/micromamba