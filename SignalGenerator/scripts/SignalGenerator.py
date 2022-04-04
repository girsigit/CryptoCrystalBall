#!/usr/bin/env python
# coding: utf-8

# Version 4.1 - 2022-04-04
# - Using model OO_1
# # Version 4.0 - 2022-03-02
# - Using strategy FF1 with direct predicted signals
# # Version 3.0
# - 2022-03-02 NOT WORKING, wrong model
# - Two-model prediciton, including daily data
# - Strategy 2022_RG_2
# # Version 2.0
# - Using the remote predictor (with model CreateModelStraightLSTMPyramidBiggerFloat_R64_E64_P64_256LB_24GLAHD_SM200_20/cp_8_06000)
# - Strategy 2022_D1
# ### Version 1.2
# - Using old algorithm again
# ### Version 1.1
# - Bugfix: Signals have to be saved without "-USDT"
# ### Version 1.0
# - Important fix: Exit has to be done without volume constraint
# - Other algorithm, using 24h Volume
# ### Version 0.3
# - Added Telegram sending
# ### Version 0.2
# - Changed entry and exit to Mittel 1
# - Volume filtering done using Market Summary

# In[ ]:

# Import custom modules
import sys
sys.path.insert(0, "../../IndicatorCalculator")
sys.path.insert(0, "../../DataStreamCreator")
from IndicatorCalculator import IndicatorCalculator, IndicatorCalculationError
import DataStreamCreator as dg

from dotenv import load_dotenv
import os
import talib
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np
import requests
import logging


# In[ ]:


logger = logging.getLogger()
logger.setLevel(logging.INFO)


# In[ ]:

# Load the environment variables
load_dotenv("../../.env")

# In[ ]:

MARKETS_URL = "https://api.bittrex.com/v3/markets"
MARKETS_SUMMARIES_URL = "https://api.bittrex.com/v3/markets/summaries"
TICK_URL_HOURLY = "https://api.bittrex.com/v3/markets/{}/candles/TRADE/HOUR_1/recent"
TICK_URL_DAILY = "https://api.bittrex.com/v3/markets/{}/candles/TRADE/DAY_1/recent"


# In[ ]:

TELEGRAM_INFO_URL = os.getenv('TELEGRAM_INFO_URL')
if TELEGRAM_INFO_URL is None:
    raise Exception("TELEGRAM_INFO_URL is None")


# In[ ]:

PREDICTOR_REST_SERVICE_URL = os.getenv('PREDICTOR_REST_SERVICE_URL')
if PREDICTOR_REST_SERVICE_URL is None:
    raise Exception("PREDICTOR_REST_SERVICE_URL is None")


# In[ ]:

SIGNAL_FOLDER = '/home/signals'
MIN_CURR_PROCESS_TIME_SECONDS = 3
VOL_LIMIT = 100000.0


# In[ ]:

ENTR_THR = 0.1
ENTR_THR2 = 0.55
EXIT_THR = 0.3
EXIT_THR2 = 0.05


# In[ ]:

SHORTSPAN = 6
MIDSPAN = 48
LONGSPAN = 120

#BATCH_SIZE = 96
# Only required if using span p values, else set to 2 for current and previous value
INDICATOR_MA_PERIOD = 2

MODEL1_X_LOOKBACK_CNT = 512
MODEL2_X_LOOKBACK_CNT = 256

# In[ ]:


# ["BULL", "BEAR", "TSLA", "ZUSD", "USDN", "BTC", "ETH", "XELS"]
IGNORE_STRINGS = os.getenv('IGNORE_STRINGS')
if IGNORE_STRINGS is None:
    raise Exception("IGNORE_STRINGS is None")
else:
    IGNORE_STRINGS = json.loads(IGNORE_STRINGS)

# # Get Market Names

# In[ ]:


# Get all markets
def GetMarketNames():
    r = requests.get(MARKETS_URL)
    j = r.json()
    markets = pd.DataFrame(j)

    # Filter the markets
    markets = markets[markets['status'] == "ONLINE"]
    markets = markets[markets['quoteCurrencySymbol'] == "USDT"]
    marketsList = markets['symbol'].values

    return marketsList


# In[ ]:


# Get all markets
def GetMarketSummaries():
    r = requests.get(MARKETS_SUMMARIES_URL)
    j = r.json()
    marketsSummaries = pd.DataFrame(j)
    marketsSummaries.set_index("symbol", inplace=True)

    marketsSummaries["quoteVolume"] = pd.to_numeric(
        marketsSummaries["quoteVolume"], downcast="float")
    marketsSummaries["percentChange"] = pd.to_numeric(
        marketsSummaries["percentChange"], downcast="float")

    # Filter the markets
#     markets = markets[markets['status'] == "ONLINE"]
#     markets = markets[markets['quoteCurrencySymbol'] == "USDT"]
#     marketsList = markets['symbol'].values

    return marketsSummaries


# # Tick getter

# In[ ]:


def GetTicks(market, tickUrl):
    u = tickUrl.format(market)
    r = requests.get(u)
    r.json()
    j = r.json()

    if 'code' in j:
        logging.error("Error in API response")
        logging.error(j)
        logging.error(u)

        return None

    # Refurbish values
    for i in range(len(j)):
        try:
            j[i]['startsAt'] = int(datetime.strptime(
                j[i]['startsAt'], '%Y-%m-%dT%H:%M:%SZ').timestamp())
            j[i]['open'] = float(j[i]['open'])
            j[i]['high'] = float(j[i]['high'])
            j[i]['low'] = float(j[i]['low'])
            j[i]['close'] = float(j[i]['close'])
            j[i]['volume'] = float(j[i]['volume'])
            j[i]['quoteVolume'] = float(j[i]['quoteVolume'])
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            logging.error(u)
            logging.error(j[i])
            logging.error(ex)
            return None

    return j


# In[ ]:


# Init IndicatorCalculator
ic = IndicatorCalculator(
    shortspan=SHORTSPAN, midspan=MIDSPAN, longspan=LONGSPAN)

# # Signals

# In[ ]:


def GenerateSignals(p1, p2, pday, quoteVolume):
    # BASE_PATH2 == short signals

    p1_entry = p1[-1, 0]
    p1_entry_previous = p1[-2, 0]

    p1_exit = p1[-1, 1]
    p1_exit_previous = p1[-2, 1]

    # p2_dir = p2[-1, 0]
    # p2_dir_previous = p2[-2, 0]

    # pday_dir = pday[-1, 0]

    # _FF_1
    _entr = (p1_entry >= ENTR_THR) & (p1_entry_previous < ENTR_THR) & (
        p1_exit <= ENTR_THR2) & (quoteVolume >= VOL_LIMIT)

    _exit = (p1_exit >= EXIT_THR) & (
        p1_exit_previous < EXIT_THR) & (p1_entry <= EXIT_THR2)

    if _entr or _exit:
        logging.info({
            'entry': _entr,
            'exit': _exit,
            'p1_entry': p1_entry,
            'p1_entry_previous': p1_entry_previous,
            'p1_exit': p1_exit,
            'p1_exit_previous': p1_exit_previous,
            'quoteVolume': quoteVolume
        })

    return {
        "entry": _entr,
        "exit": _exit
    }


# In[ ]:


def WriteSignalFile(cur, direction):
    if not os.path.exists(SIGNAL_FOLDER):
        logging.error("Signal folder %s does not exist!" % SIGNAL_FOLDER)
        return

    try:
        cur_clean = cur.replace("-USDT", "")
        fname = str(int(datetime.utcnow().timestamp())) + \
            "_" + str(direction) + "_" + cur_clean
        f = open(SIGNAL_FOLDER + os.path.sep + fname, "w")
        f.close()
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        logging.error("Exception at writing signal file for " + str(cur))
        logging.error(ex)


# WriteSignalFile('i-am-a-debug-signal', 'entry-or-exit')


# In[ ]:


def SendTelegramInfo(message):
    try:
        message = TELEGRAM_INFO_URL + datetime.utcnow().strftime("%d.%m.%Y %H:%M:%S") + \
            '\n' + message
        requests.post(message)
    except Exception as ex:
        logging.error("Exception in sending telegram message")
        logging.error(message)
        logging.error(ex)


# # Work

# In[ ]:


marketsList = GetMarketNames()
logging.info("len(marketsList): " + str(len(marketsList)))

suitable_markets = []

for m in marketsList:
    # Ignore ones
    _ignore = False
    for ign in IGNORE_STRINGS:
        if ign in m:
            _ignore = True
            break

    if not _ignore:
        suitable_markets.append(m)

logging.info("len(suitable_markets): " + str(len(suitable_markets)))


# In[ ]:


ms = GetMarketSummaries()
logging.info("len(ms): " + str(len(ms)))


# In[ ]:

def PrepareTickData(rawTickData):
    _ticksDF = pd.DataFrame(rawTickData)
    _ticksDF.set_index("startsAt", inplace=True)
    _ticksDF.sort_index(inplace=True)

    # Drop latest (instable) index
    _ticksDF = _ticksDF.iloc[:-1, :]

    # Drop Quote Volume
    _ticksDF = _ticksDF.drop("quoteVolume", axis=1)

    _indDF = ic.CreateAllIndicatorsTable(_ticksDF)
    _normedDF = ic.NormPriceRelated(_indDF)

    return _normedDF

# In[]:


def GetRemotePrediction(normedDFIn, modelNr):
    if 1 == modelNr:
        _X_LBCNT = MODEL1_X_LOOKBACK_CNT
    elif 2 == modelNr:
        _X_LBCNT = MODEL2_X_LOOKBACK_CNT
    else:
        return None

    # Plus the MA period and one before, to calc previous of MA
    normedDFSlice = normedDFIn.tail(
        _X_LBCNT + INDICATOR_MA_PERIOD + 1)

    # Block Generator
    try:
        xg = dg.XBlockGenerator(
            normedDFSlice, normedDFSlice.shape[0], _X_LBCNT)
        X = next(xg)
    except StopIteration:
        logging.warning("StopIteration occured for " + m)
        return "continue"

    # Create JSON
    j = {"modelNumber": modelNr,
         "data": X.tolist()
         }

    try:
        r = requests.post(PREDICTOR_REST_SERVICE_URL + "/predict", json=j)
    except requests.exceptions.ConnectionError as cex:
        logging.error("Connection Error to predictor", cex)
        # Todo: Send error message via telegram
        return None

    if 200 != r.status_code:
        logging.error("200 != r.status_code, it is " + str(r.status_code))

    content = r.content
    j_content = json.loads(content)

    if "error" in j_content:
        logging.error(j_content["error"])
        return None

    # Extract predictions
    p = np.array(j_content)

    return p

# In[ ]:


logging.info(datetime.now())

for m in suitable_markets:
    startTime = datetime.utcnow().timestamp()
    try:
        logging.info(m)
        ticksHourly = GetTicks(m, TICK_URL_HOURLY)
        normedDFHourly = PrepareTickData(ticksHourly)

        # Predict daily
        # ticksDaily = GetTicks(m, TICK_URL_DAILY)
        # normedDFDaily = PrepareTickData(ticksDaily)

        # Model 2 daily prediction
        # pday = GetRemotePrediction(normedDFDaily, 2)
        # if pday is None:
        #     break
        # elif isinstance(pday, str) and "continue" == pday:
        #     continue

        # Model 1 prediction
        p1 = GetRemotePrediction(normedDFHourly, 1)
        if p1 is None:
            break
        elif isinstance(p1, str) and "continue" == p1:
            continue

        # Model 2 prediction
        # p2 = GetRemotePrediction(normedDFHourly, 2)
        # if p2 is None:
        #     break
        # elif isinstance(p2, str) and "continue" == p2:
        #     continue

        # Get 24h data
        qv = ms.loc[m, "quoteVolume"]
        # pc = ms.loc[m, "percentChange"] / 100.0

        # Generate Signals
        signals = GenerateSignals(p1, None, None, qv)

        if signals['exit']:
            msg = m + ' exit signal'
            logging.info(msg)
            logging.info(signals)

            SendTelegramInfo(msg)
            WriteSignalFile(m, 'exit')
        elif signals['entry']:
            msg = m + ' entry signal'
            logging.info(msg)
            logging.info(signals)

            SendTelegramInfo(msg)
            WriteSignalFile(m, 'entry')

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        logging.error("General Exception in " + str(m))
        logging.error(ex)

    elapsed = datetime.utcnow().timestamp() - startTime
    waitTime = np.max([0.0, MIN_CURR_PROCESS_TIME_SECONDS - elapsed])
    time.sleep(waitTime)

logging.info(datetime.now())
logging.info(str(datetime.utcnow()) + " - Work is done!")
