NAME=pdf-translator
TAG=0.1.1
PROJECT_DIRECTORY=$(shell pwd)/..

build:
	docker build -t ${NAME}:${TAG} .

run:
	docker run -it \
		--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
		--name pdf-translator \
		-v ${PROJECT_DIRECTORY}:/home/${NAME} \
		--gpus all \
		-d --restart=always \
		-p 8765:8765 \
		${NAME}:${TAG} /bin/bash -c "python3 main.py"

translate:
	@cd ${PROJECT_DIRECTORY} && \
		python3 translator.py -i ${INPUT}
