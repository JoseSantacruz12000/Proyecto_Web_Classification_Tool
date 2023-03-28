#!/bin/bash
CONTAINER_NAME=mlflow_demo

if [ "$(docker ps -aq -f status=running -f name=$CONTAINER_NAME)" ]; then
  docker exec -it $CONTAINER_NAME uvicorn app.api:app --host 127.0.0.1 --port 4000
fi