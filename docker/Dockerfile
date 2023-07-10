FROM debian:11

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        gnupg2 \
        curl \
        ca-certificates \
        && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -yq --no-install-recommends --fix-missing \
        cuda-cudart-12-1=12.1.105-1 \
        cuda-compat-12-1 \
        cuda-libraries-12-1=12.1.1-1 \
        cuda-nvtx-12-1=12.1.105-1 \
        cuda-nvml-dev-12-1=12.1.105-1 \
        cuda-command-line-tools-12-1=12.1.1-1 \
        cuda-libraries-dev-12-1=12.1.1-1 \
        cuda-minimal-build-12-1=12.1.1-1 \
        libcublas-12-1=12.1.3.1-1 \
        libcublas-dev-12-1=12.1.3.1-1 \
        libcusparse-12-1=12.1.0.106-1 \
        libcusparse-dev-12-1=12.1.0.106-1 \
        libcudnn8=8.9.0.131-1+cuda12.1 \
        libcudnn8-dev=8.9.0.131-1+cuda12.1 \
        libncursesw5 \
        libtinfo5 \
        && \
    ln -s /usr/local/cuda-12.1 /usr/local/cuda && \
    find /usr/local/cuda-12.1/lib64/ -type f -name '*.a' -not -name 'libcudart_static.a' -not -name 'libcudadevrt.a' -delete && \
    rm /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/*

# CUDA environment variables
ENV CUDA_HOME "/usr/local/cuda"
ENV PATH "${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH "${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.1 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=510,driver<511 brand=unknown,driver>=510,driver<511 brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 brand=geforce,driver>=510,driver<511 brand=geforcertx,driver>=510,driver<511 brand=quadro,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511 brand=titan,driver>=510,driver<511 brand=titanrtx,driver>=510,driver<511 brand=tesla,driver>=515,driver<516 brand=unknown,driver>=515,driver<516 brand=nvidia,driver>=515,driver<516 brand=nvidiartx,driver>=515,driver<516 brand=geforce,driver>=515,driver<516 brand=geforcertx,driver>=515,driver<516 brand=quadro,driver>=515,driver<516 brand=quadrortx,driver>=515,driver<516 brand=titan,driver>=515,driver<516 brand=titanrtx,driver>=515,driver<516 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526"

RUN apt-get -q -y update && \
    apt-get install libcurl3-gnutls -q -y && \
    apt-get install -q -y bzip2 \
        wget \
        curl \
        git \
        git-lfs \
        build-essential \
        ca-certificates \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        tk-dev \
        uuid-dev \
        zlib1g-dev && \
    apt-get -y autoclean && apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get -q -y update && \
    apt-get install --no-install-recommends -yq \
        software-properties-common \
        gnupg1 \
        gnupg2 \
        ninja-build \
        unzip \
        patch \
        clang-format-13 \
        clang \
        python3-dev \
        python3-pip \
        zsh \
        openssh-client && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install --force --skip-smudge

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --no-cache-dir \
        lit \
        numpy \
        pytest \
        cmake \
        pybind11 \
        black \
        torch==2.0.1

# for onnx-frontend
RUN apt-get update && \
    apt-get install --no-install-recommends -yq \
        protobuf-compiler \
        libprotobuf-dev \
        && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        onnx==1.13.0 \
        onnxruntime==1.13.1

RUN apt-get remove -y lsb-release