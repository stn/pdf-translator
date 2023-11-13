#!/usr/bin/bash

TAG="0.2"

docker run \
  --name pdf-translator \
  --gpus all \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -u $(id -u) \
  -v $(pwd):/home/pdf-translator \
  --rm \
  pdf-translator:"${TAG}" /bin/bash -c "python3 -m pdf-translator.main $*"
