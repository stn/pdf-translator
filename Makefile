NAME=pdf-translator
TAG=0.2

.PHONY: build

build:
	docker build \
	    --build-arg UID=$(shell id -u) \
	    -t ${NAME}:${TAG} .
