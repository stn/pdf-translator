NAME=pdf-translator
TAG=0.2
PROJECT_DIRECTORY=$(shell pwd)

build:
	docker build \
	    --build-arg UID=$(shell id -u) \
	    -t ${NAME}:${TAG} .

download:
    wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/Japanese/SourceHanSerif-Light.otf

server:
	docker run \
		--shm-size=1g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name pdf-translator \
		-v ${PROJECT_DIRECTORY}:/home/pdf-translator \
		--gpus all \
		-d --restart=always \
		-p 8765:8765 \
		-u $(shell id -u) \
		${NAME}:${TAG} /bin/bash -c "python3 -m pdf-translator.server"

translate:
	@cd ${PROJECT_DIRECTORY} && \
		python3 translator.py -i ${INPUT}
