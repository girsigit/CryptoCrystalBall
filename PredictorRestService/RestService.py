import os

from flask import Flask
from flask import request

from waitress import serve

import json
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Flatten, Activation, LSTM, Dropout
from tensorflow.keras.models import Model

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_PATH = "model/CreateModelStraightLSTMPyramidBiggerFloat_R64_E64_P64_256LB_24GLAHD_SM200_20/cp_8_06000/model.h5"

BATCH_SIZE = 256
LOOKBACK_CNT = 256
FEATURES = 228
PATTERN_CNT = 61

def CreateModelStraightLSTMPyramidBiggerFloat():
  EXTRACTOR_SIZE = 128
  PYRAMID_SIZE = 64

  # Build your model input
  input = Input(shape=(LOOKBACK_CNT,FEATURES), name='input', dtype='float32')
  inputAct = Activation('tanh')(input)

  # Extractor
  extract1 = LSTM(EXTRACTOR_SIZE, return_sequences=True, name="Extractor1")(inputAct)
  extract2 = LSTM(EXTRACTOR_SIZE*2, return_sequences=True, name="Extractor2")(extract1)
  extract3 = LSTM(EXTRACTOR_SIZE*4, return_sequences=True, name="Extractor3")(extract2)
  extract4 = LSTM(EXTRACTOR_SIZE*8, return_sequences=False, name="Extractor4")(extract3)

  # Pyramid
  pyramid1 = Dense(PYRAMID_SIZE, activation="relu", name="Pyramid1")(
      Flatten(name="Flatten1")(
          Dropout(0.33, name="Dropout1")(extract1)
      )
  )
  pyramid2 = Dense(PYRAMID_SIZE, activation="relu", name="Pyramid2")(
      Flatten(name="Flatten2")(
          Dropout(0.25, name="Dropout2")(extract2)
      )
  )
  pyramid3 = Dense(PYRAMID_SIZE, activation="relu", name="Pyramid3")(
      Flatten(name="Flatten3")(
          Dropout(0.1, name="Dropout3")(extract3)
      )
  )
  pyramid4 = Dense(PYRAMID_SIZE, activation="relu", name="Pyramid4")(
      Flatten(name="Flatten4")(
          Dropout(0.05, name="Dropout4")(extract4)
      )
  )

  added = Add(name="Add")([pyramid1, pyramid2, pyramid3, pyramid4])
  postprocessor = Dense(PYRAMID_SIZE, activation="relu", name="Postprocessor1")(added)
  postprocessor = Dense(PYRAMID_SIZE, activation="relu", name="Postprocessor2")(postprocessor)
  postprocessor = Dense(PYRAMID_SIZE, activation="tanh", name="Postprocessor3")(postprocessor)
  output = Dense(2, activation='tanh', name="Output")(postprocessor)
  outputs = [output]

  # And combine it all in a model object
  model = Model(inputs=input, outputs=outputs, name='CreateModelStraightLSTMPyramidBiggerFloat')

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

    # Predict from X-Blocks
    @app.route('/predict', methods=['POST'])
    def storeLog():
        logging.info(str(datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")) +  " - predict")

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
        X_parsed = np.array(X_loaded)

        start_time = datetime.utcnow().timestamp()
        p = model.predict(X_parsed, batch_size=BATCH_SIZE)
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

    model = CreateModelStraightLSTMPyramidBiggerFloat()
    model.load_weights(MODEL_PATH)

    serve(app, host='0.0.0.0', port=5000)
