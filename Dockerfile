FROM paddlepaddle/paddle:2.5.2-gpu-cuda11.7-cudnn8.4-trt8.4

ARG USER=pdf-translator
ARG UID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y \
        git \
        libpoppler-dev \
        poppler-utils \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==1.13.0+cu117 \
    torchvision==0.14.0+cu117 \
    torchaudio==0.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install --no-cache-dir \
    fastapi[uvicorn] \
    matplotlib \
    networkx \
    opencv-python-headless \
    paddleocr \
    pdf2image \
    PyPDF2 \
    python-multipart \
    sentencepiece \
    transformers \
    uvicorn

RUN useradd -m -s /bin/bash -u $UID $USER
WORKDIR /home/$USER
USER $USER

ENV HOME=/home/$USER
