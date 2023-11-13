#!/usr/bin/env bash

NAME=pdf-translator

docker exec -it ${NAME} /bin/bash -c "python3 -m pdf-translator.translator $*"
