# syntax=docker/dockerfile:1

FROM ubuntu:focal as hexagon

# To build this container, you'll need to download manually
# hexagon_sdk_lnx_3_5_installer_00006_1.zip to the root of this
# repository.
COPY ./hexagon_sdk_lnx_3_5_installer_00006_1.zip /hexagon.zip

RUN apt-get update &&\
    apt-get install -y unzip default-jre-headless && \
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
RUN apt-get install -y curl make bash lib32z1 libncurses5 lib32ncurses-dev lsb-release && \
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


FROM ubuntu:focal as halide

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    clang-tools-12 lld llvm-12-dev libclang-12-dev liblld-12-dev \
    libpng-dev libjpeg-dev libgl-dev \
    python3.9-dev python3-numpy python3-scipy python3-imageio python3-pybind11 \
    python3-distutils \
    libopenblas-dev libeigen3-dev libatlas-base-dev \
    cmake wget git && \
    apt-get clean
RUN wget -O /usr/src/halide.tar.gz https://github.com/halide/Halide/archive/refs/tags/v13.0.4.tar.gz && \
    tar -C /usr/src -xzvf /usr/src/halide.tar.gz && \
    rm /usr/src/halide.tar.gz
RUN cd /usr/src/Halide-13.0.4 && \
    cmake -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON_BINDINGS=ON \
    -DLLVM_DIR=$LLVM_ROOT/lib/cmake/llvm -DTARGET_WEBASSEMBLY=OFF -S . -B build && \
    cmake --build build --parallel
RUN cd /usr/src/Halide-13.0.4 && \
    cmake --install ./build --prefix /halide


FROM condaforge/mambaforge:4.11.0-2 as conda

COPY environment.yml .
RUN --mount=type=cache,target=/opt/conda/pkgs mamba env create -p /env -f environment.yml 


FROM ubuntu:focal as cpu-only

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y git curl clang-12 lib32z1 libncurses5 lib32ncurses-dev && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    curl -LO http://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb && \
    dpkg -i libffi6_3.2.1-8_amd64.deb && \
    rm libffi6_3.2.1-8_amd64.deb && \
    apt-get clean

COPY --from=conda /env /env
ENV PATH=/env/bin:$PATH

COPY --from=halide /usr/src/Halide-13.0.4/python_bindings/requirements.txt /halide-reqs.txt
RUN python3 -m pip install -r /halide-reqs.txt && \
    rm /halide-reqs.txt
COPY --from=halide /halide /halide
ENV PYTHONPATH "/halide/lib/python3/site-packages:${PYTHONPATH}"
ENV LD_LIBRARY_PATH "/halide/lib:${LD_LIBRARY_PATH}"

ENV CC=/usr/bin/clang-12
ENV CLANG=/usr/bin/clang-12

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ENV PYTHONPATH ".:${PYTHONPATH}"

COPY pyproject.toml ./
COPY tests ./tests
COPY setup.py .
COPY morello ./morello
RUN python3 setup.py build_ext --inplace

COPY scripts ./scripts


FROM cpu-only as with-hexagon

COPY --from=hexagon /Hexagon_SDK /Hexagon_SDK
ENV HEXAGON_SDK_ROOT=/Hexagon_SDK/3.5.4 HEXAGON_TOOLS_ROOT=/Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07
ENV PATH=/hexagon_sdk/Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07/Tools/lib:$PATH
