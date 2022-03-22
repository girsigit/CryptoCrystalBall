import os

from flask import Flask
from flask import request

from waitress import serve

import json
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Activation, LSTM, Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model

import talib

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL1_PATH = "model/StraightCNNMultiLSTM_CF256_E128_L32_T3_512LB_LF48_EN0.6_0.6_0.0_EX-0.1_0.1_SM100_100_cp_valid_7_02250_model.h5"

MODEL1_PARAMS = {
    "BATCH_SIZE": 8,
    "X_LOOKBACK_CNT": 512,
    "FEATURES": 228,
    "CNN_FILTERS": 256,
    "EXTRACTOR_SIZE":  128,
    "LSTM_SIZE": 32,
    "FRQ_SMOOTH_PERIODS": [2, 16]
}


def frequencySmooth(inArray, smoothPeriod):
    # inArray.shape = (batchSize,timesteps,features)

    _out = np.empty(inArray.shape)

    for b in range(inArray.shape[0]):
        for f in range(inArray.shape[2]):
            _smoothed = talib.MA(inArray[b, :, f], timeperiod=smoothPeriod)
            _out[b, :-int(smoothPeriod/2), f] = _smoothed[int(smoothPeriod/2):]

            del _smoothed

    return _out

# Create frequency smoothed stack
# Shape: (batchSize, FRQ_SMOOTH_PERIODS+1, timesteps,features)


def createFrequencySmoothedStack(X_raw_in, frq_smoooth_periods):
    _frq_stack = np.empty((
        X_raw_in.shape[0],
        len(frq_smoooth_periods) + 1,
        X_raw_in.shape[1],
        X_raw_in.shape[2],
    ))

    _frq_stack[:, 0, :, :] = X_raw_in

    for i in range(len(frq_smoooth_periods)):
        _frq_stack[:, i+1, :,
                   :] = frequencySmooth(X_raw_in, frq_smoooth_periods[i])

    _frq_stack = np.nan_to_num(_frq_stack, nan=0.0, posinf=0.0, neginf=0.0)

    return _frq_stack

# @title CreateModelStraightCNNInputLSTMPyramidFloat3


# @title CreateModelStraightCNNMultiLSTM
# Inspired by https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57
def CreateModelStraightCNNMultiLSTM():
    # Build your model input
    input = Input(shape=(len(MODEL1_PARAMS['FRQ_SMOOTH_PERIODS'])+1,
                  MODEL1_PARAMS['X_LOOKBACK_CNT'], MODEL1_PARAMS['FEATURES']), name='input', dtype='float32')
    input = Activation('tanh')(input)

    fq_input = input
    fq_local_conv_output = []

    # Apply convolution to each frequency
    for i in range(len(MODEL1_PARAMS['FRQ_SMOOTH_PERIODS'])+1):
        if i == 0:
            fq = 1
        else:
            fq = MODEL1_PARAMS['FRQ_SMOOTH_PERIODS'][i-1]

        fq_slice = fq_input[:, i, :, :]

        # Convolution Block
        conved = Conv1D(MODEL1_PARAMS['CNN_FILTERS'], 7, padding="same",
                        data_format="channels_last", name="FQ{}_Conv1D_1".format(fq))(fq_slice)
        conved = Activation('tanh', name="FQ{}_tanh_1".format(fq))(conved)
        conved = MaxPooling1D(pool_size=32, data_format="channels_last",
                              name="FQ{}_MaxPooling1D_1".format(fq))(conved)
        conved = Conv1D(MODEL1_PARAMS['CNN_FILTERS']*2, 3, padding="same",
                        data_format="channels_last", name="FQ{}_Conv1D_2".format(fq))(conved)
        conved = Activation('tanh', name="FQ{}_tanh_2".format(fq))(conved)
        conved = MaxPooling1D(pool_size=3, data_format="channels_last",
                              name="FQ{}_MaxPooling1D_2".format(fq))(conved)
        conved = LSTM(MODEL1_PARAMS['LSTM_SIZE'],
                      name="FQ{}_LSTM_1".format(fq))(conved)

        fq_local_conv_output.append(conved)

    # Convolution over all
    conc = Concatenate(name="Concatenate")(fq_local_conv_output)
    # conved = Conv1D(CNN_FILTERS*4, 7, padding="same", data_format="channels_last", name="All_Conv1D_1".format(fq))(conc)
    # conved = Activation('tanh', name="All_tanh_1".format(fq))(conved)

    # regr = Flatten()(conc)
    regr = Dense(MODEL1_PARAMS['EXTRACTOR_SIZE'],
                 activation='tanh', name="Regressor1")(conc)
    regr = Dense(MODEL1_PARAMS['EXTRACTOR_SIZE'],
                 activation='tanh', name="Regressor2")(regr)
    output = Dense(3, activation='softmax', name="Output")(regr)
    outputs = [output]

    mnamesuffix = "_CF{}_E{}_L{}".format(
        MODEL1_PARAMS['CNN_FILTERS'], MODEL1_PARAMS['EXTRACTOR_SIZE'], MODEL1_PARAMS['LSTM_SIZE'])

    # And combine it all in a model object
    model = Model(inputs=input, outputs=outputs,
                  name='StraightCNNMultiLSTM'+mnamesuffix)

    return model
# model = CreateModelStraightCNNMultiLSTM()
# model.summary()

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
            # Frequency smoothing
            _X = createFrequencySmoothedStack(
                X_parsed, MODEL1_PARAMS['FRQ_SMOOTH_PERIODS'])

            p = model1.predict(
                _X, batch_size=MODEL1_PARAMS['BATCH_SIZE'])

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

    model1 = CreateModelStraightCNNMultiLSTM()
    model1.load_weights(MODEL1_PATH)

    serve(app, host='0.0.0.0', port=5000)
