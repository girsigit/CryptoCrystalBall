#!/bin/bash
FROM python:3.8.7

# Install nano and curl 
RUN apt-get update -y
RUN apt-get install nano curl gnupg wget cron tar -y

# Install TA-Lib
COPY ./ta-lib/ /ta-lib
WORKDIR /ta-lib
RUN ./configure --prefix=/usr && \
	make && \
	make install && \
	pip install Ta-Lib

WORKDIR /

# Install pips (numpy version is important, others are not compatible with talib)
# RUN pip install pandas==1.2.0 numpy==1.21.2 sklearn requests python-dotenv
RUN pip install pandas==1.2.0 numpy==1.22.1 sklearn requests python-dotenv

# Copy crontab file
ADD crontab/crontab /etc/cron.d/crontab
RUN chmod 0600 /etc/cron.d/crontab
RUN touch /var/log/cron.log

WORKDIR /home
