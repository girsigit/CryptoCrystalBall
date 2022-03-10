import os
import subprocess

from flask import Flask
from flask import request

from waitress import serve

import json
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Flatten, Activation, LSTM, Dropout, Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL1_PATH = "model/m1_r_chk_StraightCNNInputLSTMPyramidFloat5_CF256_E128_P64_T1_512LB_0GLAHD_SM500_200_cp_valid_3_00500_model.h5"
MODEL2_PATH = "model/m2_daily_g_chk_CreateModelStraightLSTMPyramidBiggerFloat_R64_E64_P64_256LB_0GLAHD_SM100_100_cp_4_04000_model.h5"
# Attention: Model path says E64, this is wrong, it is E128!

MODEL1_PARAMS = {
    "BATCH_SIZE": 128,
    "X_LOOKBACK_CNT": 512,
    "FEATURES": 228,
    "CNN_FILTERS": 256,
    "EXTRACTOR_SIZE":  128,
    "PYRAMID_SIZE": 64
}

MODEL2_PARAMS = {
    "BATCH_SIZE": 128,
    "X_LOOKBACK_CNT": 256,
    "FEATURES": 228,
    "EXTRACTOR_SIZE": 128,
    "PYRAMID_SIZE": 64
}


def CreateModelStraightCNNInputLSTMPyramidFloat5():
    # Build your model input
    input = Input(shape=(MODEL1_PARAMS['X_LOOKBACK_CNT'], MODEL1_PARAMS['FEATURES']),
                  name='input', dtype='float32')
    # input = tf.clip_by_value(input, -1.0e3, 1.0e3)
    input = Activation('tanh')(input)

    conved = Conv1D(MODEL1_PARAMS['CNN_FILTERS'], 7, padding="same",
                    data_format="channels_last", name="Conv1D_1")(input)
    conved = MaxPooling1D(
        pool_size=4, data_format="channels_last", name="MaxPooling1D_1")(conved)
    conved = Conv1D(MODEL1_PARAMS['CNN_FILTERS']/2, 7, padding="same",
                    data_format="channels_last", name="Conv1D_2")(conved)
    conved = MaxPooling1D(
        pool_size=4, data_format="channels_last", name="MaxPooling1D_2")(conved)

    # Extractor
    extract1 = LSTM(MODEL1_PARAMS['EXTRACTOR_SIZE'], return_sequences=True,
                    name="Extractor1")(conved)
    extract2 = LSTM(MODEL1_PARAMS['EXTRACTOR_SIZE']*2, return_sequences=True,
                    name="Extractor2")(extract1)
    extract3 = LSTM(MODEL1_PARAMS['EXTRACTOR_SIZE']*4, return_sequences=True,
                    name="Extractor3")(extract2)
    extract4 = LSTM(MODEL1_PARAMS['EXTRACTOR_SIZE']*8, return_sequences=True,
                    name="Extractor4")(extract3)

    # Pyramid
    pyramid1 = Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="tanh", name="Pyramid1_2")(
        Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid1_1")(
            LSTM(MODEL1_PARAMS['PYRAMID_SIZE'], name="Pyramid_LSTM_1")(
                Dropout(0.25, name="Dropout1")(extract1)
            )
        )
    )
    pyramid2 = Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="tanh", name="Pyramid2_2")(
        Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid2_1")(
            LSTM(MODEL1_PARAMS['PYRAMID_SIZE'], name="Pyramid_LSTM_2")(
                Dropout(0.10, name="Dropout2")(extract2)
            )
        )
    )
    pyramid3 = Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="tanh", name="Pyramid3_2")(
        Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid3_1")(
            LSTM(MODEL1_PARAMS['PYRAMID_SIZE'], name="Pyramid_LSTM_3")(
                Dropout(0.05, name="Dropout3")(extract3)
            )
        )
    )
    pyramid4 = Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="tanh", name="Pyramid4_2")(
        Dense(MODEL1_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid4_1")(
            LSTM(MODEL1_PARAMS['PYRAMID_SIZE'], name="Pyramid_LSTM_4")(
                Dropout(0.05, name="Dropout4")(extract4)
            )
        )
    )

    conc = Concatenate(name="Concatenate")(
        [pyramid1, pyramid2, pyramid3, pyramid4])
    regr = Dense(64, activation='tanh', name="Regressor1")(conc)
    regr = Dense(64, activation='tanh', name="Regressor2")(regr)
    output = Dense(2, activation='tanh', name="Output")(regr)
    outputs = [output]

    mnamesuffix = "_CF{}_E{}_P{}".format(
        MODEL1_PARAMS['CNN_FILTERS'],   MODEL1_PARAMS['EXTRACTOR_SIZE'], MODEL1_PARAMS['PYRAMID_SIZE'])

    # And combine it all in a model object
    model = Model(inputs=input, outputs=outputs,
                  name='StraightCNNInputLSTMPyramidFloat5'+mnamesuffix)

    return model


def CreateModelStraightLSTMPyramidBiggerFloat():
    # Build your model input
    input = Input(shape=(MODEL2_PARAMS['X_LOOKBACK_CNT'], MODEL2_PARAMS['FEATURES']),
                  name='input', dtype='float32')
    # input = tf.clip_by_value(input, -1.0e3, 1.0e3)
    input = Activation('tanh')(input)

    # Extractor
    extract1 = LSTM(MODEL2_PARAMS['EXTRACTOR_SIZE'], return_sequences=True,
                    name="Extractor1")(input)
    extract2 = LSTM(MODEL2_PARAMS['EXTRACTOR_SIZE']*2, return_sequences=True,
                    name="Extractor2")(extract1)
    extract3 = LSTM(MODEL2_PARAMS['EXTRACTOR_SIZE']*4, return_sequences=True,
                    name="Extractor3")(extract2)
    extract4 = LSTM(MODEL2_PARAMS['EXTRACTOR_SIZE']*8, return_sequences=False,
                    name="Extractor4")(extract3)

    # Pyramid
    pyramid1 = Dense(MODEL2_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid1")(
        Flatten(name="Flatten1")(
            Dropout(0.33, name="Dropout1")(extract1)
        )
    )
    pyramid2 = Dense(MODEL2_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid2")(
        Flatten(name="Flatten2")(
            Dropout(0.25, name="Dropout2")(extract2)
        )
    )
    pyramid3 = Dense(MODEL2_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid3")(
        Flatten(name="Flatten3")(
            Dropout(0.1, name="Dropout3")(extract3)
        )
    )
    pyramid4 = Dense(MODEL2_PARAMS['PYRAMID_SIZE'], activation="relu", name="Pyramid4")(
        Flatten(name="Flatten4")(
            Dropout(0.05, name="Dropout4")(extract4)
        )
    )

    added = Add(name="Add")([pyramid1, pyramid2, pyramid3, pyramid4])
    postprocessor = Dense(MODEL2_PARAMS['PYRAMID_SIZE'], activation="relu",
                          name="Postprocessor1")(added)
    postprocessor = Dense(MODEL2_PARAMS['PYRAMID_SIZE'], activation="relu",
                          name="Postprocessor2")(postprocessor)
    postprocessor = Dense(MODEL2_PARAMS['PYRAMID_SIZE'], activation="tanh",
                          name="Postprocessor3")(postprocessor)
    output = Dense(2, activation='tanh', name="Output")(postprocessor)
    outputs = [output]

    mnamesuffix = "_E{}_P{}".format(
        MODEL2_PARAMS['EXTRACTOR_SIZE'], MODEL2_PARAMS['PYRAMID_SIZE'])

    # And combine it all in a model object
    model = Model(inputs=input, outputs=outputs,
                  name='StraightLSTMPyramidBiggerFloat'+mnamesuffix)

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

    # Route to suspend the server
    @app.route('/suspend', methods=['GET'])
    def suspendServer():
        os.system("pm-suspend")

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
        elif 2 == modelNumber:
            p = model2.predict(
                X_parsed, batch_size=MODEL2_PARAMS['BATCH_SIZE'])

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

    model1 = CreateModelStraightCNNInputLSTMPyramidFloat5()
    model1.load_weights(MODEL1_PATH)

    model2 = CreateModelStraightLSTMPyramidBiggerFloat()
    model2.load_weights(MODEL2_PATH)

    serve(app, host='0.0.0.0', port=5000)
