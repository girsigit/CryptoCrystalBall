#!/bin/bash
CONTAINER_NAME=signal-generator-predictor
CONTAINER_PATH=$PWD

IMAGE_NAME=${CONTAINER_NAME}image

# Stop and remove the old container
docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# Build the new one
docker build -t ${IMAGE_NAME} .

# Start it
docker run \
--name ${CONTAINER_NAME} \
-v ${CONTAINER_PATH}/RestService.py:/home/scripts/RestService.py \
-v ${CONTAINER_PATH}/model:/home/scripts/model \
--cpus="0.75" \
-p 6661:5000 \
-d \
--restart unless-stopped \
${IMAGE_NAME}
