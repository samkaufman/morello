# syntax=docker/dockerfile:1

FROM ubuntu:jammy AS base
ARG HALIDE_CMAKE_PARALLEL

# Set apt to use a local mirror
# RUN sed -i -e 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//mirror:\/\/mirrors\.ubuntu\.com\/mirrors\.txt/' /etc/apt/sources.list

# Python 3.10 will be useful for all of the following image layers.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3.10-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


FROM base as hexagon

# To build this container, you'll need to download manually
# hexagon_sdk_lnx_3_5_installer_00006_1.zip to the root of this
# repository.
COPY ./hexagon_sdk_lnx_3_5_installer_00006_1.zip /hexagon.zip

RUN apt-get update &&\
    apt-get install -y --no-install-recommends unzip default-jre-headless && \
    apt-get clean
RUN unzip /hexagon.zip && \
    rm /hexagon.zip && \
    tar xf Hexagon_SDK_LNX_3_5_Installer_00006_1.tar && \
    rm Hexagon_SDK_LNX_3_5_Installer_00006_1.tar && \
    chmod ugo+x qualcomm_hexagon_sdk_3_5_4_eval.bin && \
    sh qualcomm_hexagon_sdk_3_5_4_eval.bin \
    -i silent -DDOWNLOAD_ECLIPSE=false \
    -DUSER_INSTALL_DIR=/ && \
    rm -rf qualcomm_hexagon_sdk_3_5_4_eval.bin \
    /Hexagon_SDK/3.5.4/Uninstall_Hexagon_SDK \
    /Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07/Examples \
    /Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07/Documents \
    /Hexagon_SDK/3.5.4/tools/hexagon_ide \
    /Hexagon_SDK/3.5.4/tools/Uninstall_Hexagon_SDK \
    /Hexagon_SDK/3.5.4/tools/HALIDE_Tools \
    /Hexagon_SDK/3.5.4/tools/android-ndk-*

# Install libffi6, which is needed by qaic in hexagon_nn, but not in
# Ubuntu 20 (focal).
RUN apt-get install -y --no-install-recommends \
    curl make bash lib32z1 libncurses5 lib32ncurses-dev lsb-release && \
    curl -LO http://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb && \
    dpkg -i libffi6_3.2.1-8_amd64.deb && \
    rm libffi6_3.2.1-8_amd64.deb && \
    apt-get clean

RUN /bin/bash -c "cd /Hexagon_SDK/3.5.4 && \
    source ./setup_sdk_env.source && \
    cd libs/hexagon_nn/2.10.1 && \
    make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv83_v66 V66=1 && \
    cd /Hexagon_SDK/3.5.4/libs/common/qprintf && \
    make tree VERBOSE=1 V=hexagon_Release_dynamic_toolv83_v66 V66=1"


FROM base as halide

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
      apt-get install -y --no-install-recommends \
      clang-12 clang++-12 clang-tools-12 lld \
      llvm-12-dev libclang-12-dev liblld-12-dev \
      libpng-dev libjpeg-dev libgl-dev \
      libopenblas-dev libeigen3-dev libatlas-base-dev \
      cmake make wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q -O /usr/src/halide.tar.gz https://github.com/halide/Halide/archive/refs/tags/v13.0.4.tar.gz && \
    tar -C /usr/src -xzvf /usr/src/halide.tar.gz && \
    rm /usr/src/halide.tar.gz
RUN cd /usr/src/Halide-13.0.4 && \
    cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE=/usr/bin/python3.10 \
    -DCMAKE_C_COMPILER=/usr/bin/clang-12 \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++-12 \
    -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" -DTARGET_WEBASSEMBLY=OFF -S . -B build && \
    cmake --build build --parallel ${HALIDE_CMAKE_PARALLEL}
RUN cd /usr/src/Halide-13.0.4 && \
    cmake --install ./build --prefix /halide


FROM base as poetry

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.4.1 \
    POETRY_VENV=/opt/poetry-venv

# Install poetry in its own venv
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-venv \
    && python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install "poetry==${POETRY_VERSION}" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Have Poetry pipe specific pkg. versions in pip to install.
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true \
    && python -m venv /env \
    && poetry install -n --no-ansi --no-root --with=dev,evaluation


FROM base AS cpu-only

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y $(apt-cache depends linux-tools-generic | grep Depends | sed "s/.*ends:\ //" | tr '\n' ' ') && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      git curl lib32z1 libncurses5 lib32ncurses-dev numactl \
      clang-14 lld libomp5-14 libomp-14-dev && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    curl -LO http://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb && \
    dpkg -i libffi6_3.2.1-8_amd64.deb && \
    rm libffi6_3.2.1-8_amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=poetry /.venv /.venv
ENV PATH="/.venv/bin:$PATH" \
    CLANG=/usr/bin/clang-14 \
    CC=/usr/bin/clang-14 \
    CXX=/usr/bin/clang++-14 \
    LD_LIBRARY_PATH="/.venv/lib:$LD_LIBRARY_PATH" \
    LIBRARY_PATH="/.venv/lib:$LIBRARY_PATH"

COPY --from=halide /usr/src/Halide-13.0.4/python_bindings/requirements.txt /halide-reqs.txt
RUN python -m pip install -r /halide-reqs.txt \
    && rm /halide-reqs.txt
COPY --from=halide /halide /halide
ENV PYTHONPATH="/halide/lib/python3/site-packages:${PYTHONPATH}" \
    LD_LIBRARY_PATH="/halide/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

ENV PYTHONFAULTHANDLER=1 \
    PYTHONPATH=".:${PYTHONPATH}" \
    MORELLO_CLANG_LINK_RT=1

COPY pyproject.toml setup.py ./
COPY tests ./tests
COPY morello ./morello
# RUN CC=clang-14 LDSHARED="clang-14 -shared" python3 setup.py build_ext -j "$(nproc)" --inplace
COPY scripts ./scripts


FROM cpu-only as with-hexagon

COPY --from=hexagon /Hexagon_SDK /Hexagon_SDK
ENV HEXAGON_SDK_ROOT=/Hexagon_SDK/3.5.4 HEXAGON_TOOLS_ROOT=/Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07
ENV PATH=/hexagon_sdk/Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07/Tools/lib:$PATH
