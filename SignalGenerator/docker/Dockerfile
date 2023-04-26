# Start from python
FROM python:3.10.9

# Install TF without gpu (fast enough for prediction), version 2.10.0
RUN pip install tensorflow==2.10.0

# Switch user to root for installations and build of Ta-Lib
#USER root

#RUN uname -m && cat /etc/os-release
#RUN stop

# Install the basic nvidia libs
# RUN apt -y update && apt -y install software-properties-common

# RUN wget https://developer.download.nvidia.com/compute/cuda/11.5.1/local_installers/cuda-repo-debian11-11-5-local_11.5.1-495.29.05-1_amd64.deb
# RUN dpkg -i cuda-repo-debian11-11-5-local_11.5.1-495.29.05-1_amd64.deb
# RUN apt-key add /var/cuda-repo-debian11-11-5-local/7fa2af80.pub
# RUN add-apt-repository contrib
# RUN apt-get update
# RUN apt-get -y install cuda

# https://docs.nvidia.com/cuda/pdf/CUDA_Installation_Guide_Linux.pdf, Chapter 5.11
# RUN add-apt-repository contrib && apt-key del 7fa2af80
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb

# RUN apt -y update && apt -y install nvidia-cuda-toolkit
# RUN add-apt-repository ppa:graphics-drivers/ppa && apt -y update && apt -y install nvidia-compute-utils-510 && apt -y install nvidia-cuda-toolkit

#RUN add-apt-repository ppa:graphics-drivers/ppa && apt -y update && apt -y install nvidia-compute-utils-510 && apt -y install nvidia-cuda-toolkit

# Install the CUDNN framework
# Todo: Provide cloud storage links for these files
#COPY libcudnn-setup/*.deb /
#RUN apt install /libcudnn8_8.7.0.84-1+cuda11.8_amd64.deb && apt install /libcudnn8-dev_8.7.0.84-1+cuda11.8_amd64.deb && apt install /libcudnn8-samples_8.7.0.84-1+cuda11.8_amd64.deb

# Install gsutils
#RUN apt -y install apt-transport-https ca-certificates gnupg curl && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && apt -y update && apt -y install google-cloud-cli

# Build the Ta-Lib library
WORKDIR /talib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzvf ta-lib-0.4.0-src.tar.gz
WORKDIR /talib/ta-lib
RUN ./configure --prefix=/usr && make && make install

# Set the workdir to '/content' to be consistent with Google Colab
WORKDIR /content

# Change the owner of the 'lab' and 'content' folder, by default its root and therefore not writeable
#RUN mkdir /home/jovyan/.jupyter/lab && chown jovyan /home/jovyan/.jupyter/lab && chown jovyan /content

# Install python libs
RUN pip install Ta-Lib transformers==4.22 tensorboard-plugin-profile pandas scikit-learn

# Add a non-root user and switch to it
# https://stackoverflow.com/a/70520801
RUN adduser --system --no-create-home nonroot
USER nonroot

# Prepare it for the connection to Google Colab to use their fancy UI with the local runtime
#RUN pip install jupyter_http_over_ws && jupyter serverextension enable --py jupyter_http_over_ws

# Create port expose and start command
# This command includes also the permission to connect to the local runtime from a Colab instance
ENTRYPOINT ["python3"]



#!/bin/bash


# FROM python:3.8.7

# # Install nano and curl 
# RUN apt-get update -y
# RUN apt-get install nano curl gnupg wget cron tar -y

# # Install TA-Lib
# COPY ./ta-lib/ /ta-lib
# WORKDIR /ta-lib
# RUN ./configure --prefix=/usr && \
# 	make && \
# 	make install && \
# 	pip install Ta-Lib

# WORKDIR /

# # Install pips (numpy version is important, others are not compatible with talib)
# # RUN pip install pandas==1.2.0 numpy==1.21.2 sklearn requests python-dotenv
# RUN pip install pandas==1.2.0 numpy==1.22.1 sklearn requests python-dotenv

# # Copy crontab file
# ADD crontab/crontab /etc/cron.d/crontab
# RUN chmod 0600 /etc/cron.d/crontab
# RUN touch /var/log/cron.log

# WORKDIR /home
