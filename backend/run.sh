#!/bin/bash
IMAGE_NAME=backend/app
CONTAINER_NAME=backend_container

docker build -t $IMAGE_NAME .
docker run --name $CONTAINER_NAME  --user root -e GRANT_SUDO=yes --rm -p 8888:8888 -p 4000:4000 -it $IMAGE_NAME