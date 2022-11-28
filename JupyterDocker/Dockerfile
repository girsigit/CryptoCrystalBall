# Start from a Jupyter notebook container with Tensorflow
FROM jupyter/tensorflow-notebook

# Switch user to root to build Ta-Lib
USER root

# Build the Ta-Lib library
WORKDIR /talib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && tar -xzvf ta-lib-0.4.0-src.tar.gz
  # !unzip /content/drive/MyDrive/Privat/Crypto/talibmaked.zip -d /
WORKDIR /talib/ta-lib
RUN ./configure --prefix=/usr && make && make install

# Switch back to normal user to install Ta-Lib
USER jovyan
RUN pip install Ta-Lib

# Set the workdir to the home dir of the user
WORKDIR /home/jovyan

#COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

# Create port expose and start command
EXPOSE 8888
ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]

