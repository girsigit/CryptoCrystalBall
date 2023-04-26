# Todo: Write instructions how to get the .deb files and how to install them


### Install the CUDNN framework
Change the paths in the Dockerfile

COPY *.deb /
RUN apt install /libcudnn8_8.7.0.84-1+cuda11.8_amd64.deb && apt install /libcudnn8-dev_8.7.0.84-1+cuda11.8_amd64.deb && apt install /libcudnn8-samples_8.7.0.84-1+cuda11.8_amd64.deb
