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


FROM teeks99/clang-ubuntu@sha256:8caa3a9c5c904dc276e52275ee74df57d6b873c6fa2ef7e8f4bc15b59c74efb7
COPY --from=hexagon /Hexagon_SDK /Hexagon_SDK
ENV HEXAGON_SDK_ROOT=/Hexagon_SDK/3.5.4 HEXAGON_TOOLS_ROOT=/Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07

RUN apt-get update && \
        apt-get install -y curl python3.9 python3.9-distutils lib32z1 libncurses5 lib32ncurses-dev  && \
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
        python3.9 get-pip.py && \
        rm get-pip.py && \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
        curl -LO http://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb && \
        dpkg -i libffi6_3.2.1-8_amd64.deb && \
        rm libffi6_3.2.1-8_amd64.deb && \
        apt-get clean


WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip install -U pytest-xdist
COPY pyproject.toml ./
COPY tests ./tests

COPY morello ./morello
COPY scripts ./scripts

ENV PATH=/hexagon_sdk/Hexagon_SDK/3.5.4/tools/HEXAGON_Tools/8.3.07/Tools/lib:$PATH
ENV PYTHONPATH "."
ENV CLANG=/usr/bin/clang-12
