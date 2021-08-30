# syntax=docker/dockerfile:1
# FROM python:3.9-slim-buster
FROM silkeh/clang:12

RUN apt-get update && \
        apt-get install -y python3 python3-pip && \
        apt-get clean

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY morello ./morello
COPY scripts ./scripts
ENV PYTHONPATH "."
