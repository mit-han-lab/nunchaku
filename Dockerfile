# Use an NVIDIA base image with CUDA support
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /

# Setup timezone and install system dependencies
RUN 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select New_York' | debconf-set-selections \
    && apt update -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y && apt update \
    && apt install python3.11 python3.11-dev g++-11 gcc-11 -y \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 && apt install python3.11-distutils -y \
    && apt install curl git sudo libibverbs-dev -y \
    && apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && update-alternatives --set python /usr/bin/python3.11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 1 && update-alternatives --set gcc /usr/bin/gcc-11 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 1 && update-alternatives --set g++ /usr/bin/g++-11 \
    && apt clean

# Install building dependencies
RUN pip install torch torchvision torchaudio \
    && pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub comfy-cli

# Start building
RUN git clone https://github.com/mit-han-lab/nunchaku.git \
    && cd nunchaku \
    && git submodule init \
    && git submodule update \
    && NUNCHAKU_INSTALL_MODE=ALL pip install -e . --no-build-isolation

RUN cd .. && git clone https://github.com/comfyanonymous/ComfyUI \
    && cd ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager \
    && git clone https://github.com/mit-han-lab/ComfyUI-nunchaku.git nunchaku_nodes \
    && cd .. && mkdir -p user/default/workflows/ && cp custom_nodes/nunchaku_nodes/workflows/* user/default/workflows/

## Set default command
#CMD ["python3", "app.py"]
