from threading import Thread
from wakeonlan import send_magic_packet
import os
from dotenv import load_dotenv
import requests
import time

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load the environment variables
load_dotenv("../.env")

PREDICTOR_REST_SERVICE_MAC = os.getenv('PREDICTOR_REST_SERVICE_MAC')
if PREDICTOR_REST_SERVICE_MAC is None:
    raise Exception("PREDICTOR_REST_SERVICE_MAC is None")

PREDICTOR_REST_SERVICE_URL = os.getenv('PREDICTOR_REST_SERVICE_URL')
if PREDICTOR_REST_SERVICE_URL is None:
    raise Exception("PREDICTOR_REST_SERVICE_URL is None")


RETRY_CNT = 20
RETRY_TIMEOUT_SECONDS_INITIAL = 3
RETRY_TIMEOUT_SECONDS = 1


for current_try in range(RETRY_CNT + 1):
    if RETRY_CNT == current_try:
        # Todo: Telegram error message
        raise Exception("Prediction server not responding after " + str(RETRY_CNT) + " tries")

    # Wake up server
    send_magic_packet(PREDICTOR_REST_SERVICE_MAC)

    # Wait to settle
    if 0 == current_try:
        time.sleep(RETRY_TIMEOUT_SECONDS_INITIAL)
    else:
        time.sleep(RETRY_TIMEOUT_SECONDS)

    # Check if the server is alive
    try:
        r = requests.get(PREDICTOR_REST_SERVICE_URL + "/alive")

        if 200 == r.status_code:
            logging.info("Prediction server responded alive at try " + str(current_try))
            break
        else:
            logging.warning("Check if prediction server is running failed with http code " + str(r.status_code))
    except requests.exceptions.ConnectionError as cex:
        logging.warning("Check if prediction server is running failed with ConnectionError: " + str(cex))
    
print("Finished")