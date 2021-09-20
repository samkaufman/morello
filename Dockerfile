# syntax=docker/dockerfile:1

FROM debian:bullseye as hexagon

# To build this container, you'll need to download manually
# hexagon_sdk_lnx_3_5_installer_eval_00005_3.zip to the root of this
# repository.
COPY ./hexagon_sdk_lnx_3_5_installer_eval_00005_3.zip /hexagon.zip

RUN apt-get update &&\
        apt-get install -y unzip default-jre-headless && \
        apt-get clean
RUN unzip /hexagon.zip && \
        rm /hexagon.zip && \
        tar xf Hexagon_SDK_LNX_3_5_Installer_Eval_00005_3.tar && \
        rm Hexagon_SDK_LNX_3_5_Installer_Eval_00005_3.tar && \
        chmod ugo+x qualcomm_hexagon_sdk_3_5_3_eval.bin && \
        sh qualcomm_hexagon_sdk_3_5_3_eval.bin \
        -i silent -DDOWNLOAD_ECLIPSE=false \
        -DUSER_INSTALL_DIR=/ && \
        rm qualcomm_hexagon_sdk_3_5_3_eval.bin && \
        rm -rf /Hexagon_SDK/3.5.3/tools/hexagon_ide && \
        rm -rf /Hexagon_SDK/3.5.3/tools/Uninstall_Hexagon_SDK 


FROM silkeh/clang:12
COPY --from=hexagon /Hexagon_SDK /Hexagon_SDK
ENV HEXAGON_SDK_ROOT=/Hexagon_SDK/3.5.3 HEXAGON_TOOLS_ROOT=/Hexagon_SDK/3.5.3/tools/HEXAGON_Tools/8.3.07

RUN apt-get update && \
        apt-get install -y python3 python3-pip libncurses5 && \
        apt-get clean


WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN pip install -U pytest-xdist
COPY pyproject.toml ./
COPY pytest.ini ./
COPY tests ./tests

COPY morello ./morello
COPY scripts ./scripts

ENV PATH=/hexagon_sdk/Hexagon_SDK/3.5.3/tools/HEXAGON_Tools/8.3.07/Tools/lib:$PATH
ENV PYTHONPATH "."
