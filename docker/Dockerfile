FROM paddlepaddle/paddle:2.5.1-gpu-cuda11.7-cudnn8.4-trt8.4

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get install -y \
        poppler-utils \
        libpoppler-dev \
        wget git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/pdf-translator

WORKDIR /home/pdf-translator

RUN pip install --no-cache-dir torch==1.13.0+cu117 \
    torchvision==0.14.0+cu117 \
    torchaudio==0.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip install --no-cache-dir \
    opencv-python-headless \
    networkx \
    transformers \
    sentencepiece \
    paddleocr \
    pdf2image \
    uvicorn \
    python-multipart \
    fastapi[uvicorn] \
    PyPDF2 \
    matplotlib

RUN wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/Japanese/SourceHanSerif-Light.otf -P /home/
