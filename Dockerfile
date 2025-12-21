FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.9;8.6;8.0;9.0" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps in one layer + cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1-mesa-glx \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (cached downloads)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

# Python deps in fewer layers (cached downloads)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      pyhocon open3d scipy pandas trimesh \
      PyMCubes \
      libigl \
      opencv-python \
      python-igraph \
      spconv-cu126 \
      yacs h5py tensorboardX \
      comet_ml \
      py-cpuinfo \
      timm \
      einops \
      "pytorch-lightning==2.6.0" \
      "segmentation-models-pytorch==0.4.0" \
      "monai==1.4.0"\
      "scikit-image==0.25.0"

# Git installs (kept separate so they only rebuild if these lines change)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
      "git+https://github.com/cong-yi/DualMesh-UDF" \
      "git+https://github.com/sbarratt/torch_interpolations.git"

