NAME=pdf-translator
TAG=0.2
PROJECT_DIRECTORY=$(shell pwd)

.PHONY: server

assets: SourceHanSerif-Light.otf
    wget https://github.com/adobe-fonts/source-han-serif/raw/release/OTF/Japanese/SourceHanSerif-Light.otf

build: assets
	docker build \
	    --build-arg UID=$(shell id -u) \
	    -t ${NAME}:${TAG} .

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
		${NAME}:${TAG} python3 -m pdf-translator.server
