# Start from a Jupyter notebook container with Tensorflow
FROM jupyter/tensorflow-notebook

# Change to TF with gpu, version 2.10.0
RUN pip uninstall -y tensorflow
RUN pip install tensorflow-gpu==2.10.0

# Switch user to root for installations and build of Ta-Lib
USER root

# Install the basic nvidia libs
RUN apt -y update && apt -y install nvidia-compute-utils-510 && apt -y install nvidia-cuda-toolkit

# Install the CUDNN framework
COPY libcudnn-setup/*.deb /
RUN apt install /libcudnn8_8.7.0.84-1+cuda11.8_amd64.deb && apt install /libcudnn8-dev_8.7.0.84-1+cuda11.8_amd64.deb && apt install /libcudnn8-samples_8.7.0.84-1+cuda11.8_amd64.deb

# Install gsutils
RUN apt -y install apt-transport-https ca-certificates gnupg curl && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && apt -y update && apt -y install google-cloud-cli

# Build the Ta-Lib library
WORKDIR /talib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzvf ta-lib-0.4.0-src.tar.gz
WORKDIR /talib/ta-lib
RUN ./configure --prefix=/usr && make && make install

# Set the workdir to '/content' to be consistent with Google Colab
WORKDIR /content

# Change the owner of the 'lab' and 'content' folder, by default its root and therefore not writeable
RUN mkdir /home/jovyan/.jupyter/lab && chown jovyan /home/jovyan/.jupyter/lab && chown jovyan /content

# Switch back to normal user to install Ta-Lib
USER jovyan

# Install python libs
RUN pip install Ta-Lib transformers==4.22 tensorboard-plugin-profile

# Prepare it for the connection to Google Colab to use their fancy UI with the local runtime
RUN pip install jupyter_http_over_ws && jupyter serverextension enable --py jupyter_http_over_ws

# Create port expose and start command
# This command includes also the permission to connect to the local runtime from a Colab instance
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.allow_origin='*'", "--port=8888", "--NotebookApp.port_retries=0"]

