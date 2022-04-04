# Version 4.1 - 2022-04-04
# - Using model OO_1
# - Added support for (non-GPU) docker container

import os

from flask import Flask
from flask import request

from waitress import serve

import json
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv1D, MaxPooling1D, Concatenate, AveragePooling1D, MultiHeadAttention
from tensorflow.keras.models import Model

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL1_PATH = "model/StraightCNNMMHAMoreSteps_CF256_E128_PS16_T3_512LB_LF48_EN0.6_0.6_0.0_EX-0.1_0.1_SM48_48_newperiods_cp_valid_17_01000_model.h5"

MODEL1_PARAMS = {
    "BATCH_SIZE": 8,
    "X_LOOKBACK_CNT": 512,
    "FEATURES": 228,
    "CNN_FILTERS": 256,
    "EXTRACTOR_SIZE":  128,
    "POOL_SIZE": 16,
    "FRQ_SMOOTH_PERIODS": [1, 2, 16],
    "TIME_WARP_FACTORS": [1, 4, 16]
}

# @title CreateModelStraightCNNMMHAMoreSteps
# Inspired by https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57


def CreateModelStraightCNNMMHAMoreSteps():

    # Build your model input
    input = Input(shape=(MODEL1_PARAMS['X_LOOKBACK_CNT'],
                  MODEL1_PARAMS['FEATURES']), name='input', dtype='float32')
    input = Activation('tanh')(input)

    fq_input = input
    fq_local_conv_output = []

    # Apply convolution to each frequency
    for fq in MODEL1_PARAMS['FRQ_SMOOTH_PERIODS']:
        for tw in MODEL1_PARAMS['TIME_WARP_FACTORS']:
            fq_slice = fq_input[:, ::tw, :]

            # Averaging in time dimension --> High frequency filtering
            hffiltered = AveragePooling1D(
                pool_size=fq, strides=1, padding="same", name="FQ{}_TW{}_AvgPool".format(fq, tw))(fq_slice)

            # Convolution Block
            conved = Conv1D(MODEL1_PARAMS['CNN_FILTERS'], 11, padding="same",
                            data_format="channels_last", name="FQ{}_TW{}_Conv1D_1".format(fq, tw))(hffiltered)
            conved = MaxPooling1D(pool_size=int(
                MODEL1_PARAMS['POOL_SIZE']/tw), data_format="channels_last", name="FQ{}_TW{}_MaxPooling1D_1".format(fq, tw))(conved)
            conved = Conv1D(int(MODEL1_PARAMS['CNN_FILTERS']/2), 3, padding="same",
                            data_format="channels_last", name="FQ{}_TW{}_Conv1DReduce_1".format(fq, tw))(conved)

            conved = Conv1D(MODEL1_PARAMS['CNN_FILTERS']*2, 5, padding="same",
                            data_format="channels_last", name="FQ{}_TW{}_Conv1D_2".format(fq, tw))(conved)
            conved = MaxPooling1D(pool_size=int(
                MODEL1_PARAMS['POOL_SIZE']/tw), data_format="channels_last", name="FQ{}_TW{}_MaxPooling1D_2".format(fq, tw))(conved)
            conved = Conv1D(int(MODEL1_PARAMS['CNN_FILTERS']/2), 3, padding="same",
                            data_format="channels_last", name="FQ{}_TW{}_Conv1DReduce_2".format(fq, tw))(conved)

            conved = Conv1D(MODEL1_PARAMS['CNN_FILTERS']*4, 3, padding="same",
                            data_format="channels_last", name="FQ{}_TW{}_Conv1D_3".format(fq, tw))(conved)
            conved = MaxPooling1D(pool_size=2, data_format="channels_last",
                                  name="FQ{}_TW{}_MaxPooling1D_3".format(fq, tw))(conved)
            conved = Conv1D(int(MODEL1_PARAMS['CNN_FILTERS']/2), 3, padding="same",
                            data_format="channels_last", name="FQ{}_TW{}_Conv1DReduce_3".format(fq, tw))(conved)

            mhaSingle = MultiHeadAttention(
                num_heads=8, key_dim=2, name="FQ{}_TW{}_MHA".format(fq, tw))(conved, conved)

            fq_local_conv_output.append(mhaSingle)

    # Conc all
    conc = Concatenate(name="Concatenate", axis=1)(fq_local_conv_output)

    regr = Dense(MODEL1_PARAMS['EXTRACTOR_SIZE'],
                 activation='tanh', name="Regressor1")(conc)
    regr = Flatten(name="Regressor_Flatten")(regr)
    regr = Dense(MODEL1_PARAMS['EXTRACTOR_SIZE'],
                 activation='tanh', name="Regressor2")(regr)
    regr = Dense(MODEL1_PARAMS['EXTRACTOR_SIZE'],
                 activation='tanh', name="Regressor3")(regr)
    output = Dense(3, activation='softmax', name="Output")(regr)
    outputs = [output]

    mnamesuffix = "_CF{}_E{}_PS{}".format(
        MODEL1_PARAMS['CNN_FILTERS'], MODEL1_PARAMS['EXTRACTOR_SIZE'], MODEL1_PARAMS['POOL_SIZE'])

    # And combine it all in a model object
    model = Model(inputs=input, outputs=outputs,
                  name='StraightCNNMMHAMoreSteps'+mnamesuffix)

    return model

# Server app


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        # DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Route to check if the server is alive and ready
    @app.route('/alive', methods=['GET'])
    def returnAlive():
        response = app.response_class(
            status=200,
            mimetype='application/json'
        )
        return response

    # Predict from X-Blocks
    @app.route('/predict', methods=['POST'])
    def storeLog():
        logging.info(str(datetime.utcnow().strftime(
            "%Y-%m-%d_%H%M%S")) + " - predict")

        # Check if json payload is present
        if request.json is None:
            err_str = "REST predictor - predict request.json is None"
            logging.error(err_str)

            response = app.response_class(
                response=json.dumps({"error": err_str}),
                status=510,
                mimetype='application/json'
            )
            return response

        # Get json from request
        j = request.json

        # Extract X values from json
        X_loaded = j
        X_parsed = np.array(X_loaded['data'])
        modelNumber = int(X_loaded['modelNumber'])

        logging.info("Model " + str(modelNumber))

        start_time = datetime.utcnow().timestamp()
        if 1 == modelNumber:
            p = model1.predict(
                X_parsed, batch_size=MODEL1_PARAMS['BATCH_SIZE'])

        elapsed = datetime.utcnow().timestamp() - start_time
        logging.info("Time elapsed predicting: " + str(elapsed))

        response = app.response_class(
            response=json.dumps(p.tolist()),
            status=200,
            mimetype='application/json'
        )
        return response

    return app


if __name__ == '__main__':
    app = create_app()

    model1 = CreateModelStraightCNNMMHAMoreSteps()
    model1.load_weights(MODEL1_PATH)

    serve(app, host='0.0.0.0', port=5000)
