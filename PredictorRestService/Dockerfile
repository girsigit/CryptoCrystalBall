FROM tensorflow/tensorflow:2.7.0

RUN pip install flask waitress

WORKDIR /home/scripts

ENTRYPOINT [ "python3" ]
CMD [ "/home/scripts/RestService.py" ]