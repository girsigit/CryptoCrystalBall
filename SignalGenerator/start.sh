#!/bin/bash
CONTAINER_NAME=signal-generator
CONTAINER_PATH=$PWD

IMAGE_NAME=${CONTAINER_NAME}image

SIGNALS_PATH=/home/berni/server/signals

# Create log directories
mkdir cronlogs
mkdir cronlogs/log
mkdir cronlogs/err

# Unpack talib
tar -xf $PWD/../talib/ta-lib-0.4.0-src.tar.gz --directory $PWD/docker/

# Stop and remove the old container
docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# Build the new one
docker build -t ${IMAGE_NAME} ./docker 

# Start it
docker run \
--name ${CONTAINER_NAME} \
-v ${CONTAINER_PATH}/scripts:/home/scripts \
-v ${CONTAINER_PATH}/cronlogs:/home/cronlogs \
-v ${CONTAINER_PATH}/../DataStreamCreator:/DataStreamCreator \
-v ${CONTAINER_PATH}/../IndicatorCalculator:/IndicatorCalculator \
-v ${CONTAINER_PATH}/../.env:/.env \
-v ${SIGNALS_PATH}:/home/signals \
--cpus="0.5" \
-d \
--restart unless-stopped \
${IMAGE_NAME} \
cron -f
