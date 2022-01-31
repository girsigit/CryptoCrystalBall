#!/usr/bin/env python
# coding: utf-8

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


import requests
import numpy as np
import pandas as pd
import json
from datetime import datetime
import time
import talib
import os
import json
from dotenv import load_dotenv


# In[ ]:


# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Add, Flatten, Concatenate, Activation, LSTM, Permute
# from tensorflow.keras.models import Model

# print(tf.__version__)

# In[ ]:


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# In[ ]:


# Import custom modules
import sys

sys.path.insert(0, "../../IndicatorCalculator")
sys.path.insert(0, "../../DataStreamCreator")

from IndicatorCalculator import IndicatorCalculator, IndicatorCalculationError
import DataStreamCreator as dg


# In[ ]:

# Load the environment variables
load_dotenv("../../.env")

# In[ ]:

MARKETS_URL = "https://api.bittrex.com/v3/markets"
MARKETS_SUMMARIES_URL = "https://api.bittrex.com/v3/markets/summaries"
TICK_URL = "https://api.bittrex.com/v3/markets/{}/candles/TRADE/HOUR_1/recent"


# In[ ]:

TELEGRAM_INFO_URL = os.getenv('TELEGRAM_INFO_URL')
if TELEGRAM_INFO_URL is None:
    raise Exception("TELEGRAM_INFO_URL is None")


# In[ ]:

PREDICTOR_REST_SERVICE_URL = os.getenv('PREDICTOR_REST_SERVICE_URL')
if PREDICTOR_REST_SERVICE_URL is None:
    raise Exception("PREDICTOR_REST_SERVICE_URL is None")


# In[ ]:

SIGNAL_FOLDER = 'signals'
MIN_CURR_PROCESS_TIME_SECONDS = 2
VOL_LIMIT = 200000.0


# In[ ]:

# Todo: Check
ENTR_THR = -0.2
ENTR_THR2 = 0.04
EXIT_THR = 0.9


# In[ ]:


#BATCH_SIZE = 96
INDICATOR_MA_PERIOD = 24
LOOKBACK_CNT = 256
FEATURES = 228
PATTERN_CNT = 61


# In[ ]:


IGNORE_STRINGS = os.getenv('IGNORE_STRINGS') #["BULL", "BEAR", "TSLA", "ZUSD", "USDN", "BTC", "ETH", "XELS"]
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
    marketsSummaries.set_index("symbol",inplace=True)
    
    marketsSummaries["quoteVolume"] = pd.to_numeric(marketsSummaries["quoteVolume"], downcast="float")
    marketsSummaries["percentChange"] = pd.to_numeric(marketsSummaries["percentChange"], downcast="float")
    
    # Filter the markets
#     markets = markets[markets['status'] == "ONLINE"]
#     markets = markets[markets['quoteCurrencySymbol'] == "USDT"]
#     marketsList = markets['symbol'].values
        
    return marketsSummaries


# # Tick getter

# In[ ]:


def GetTicks(market):
    u = TICK_URL.format(market)
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
            j[i]['startsAt'] = int(datetime.strptime(j[i]['startsAt'], '%Y-%m-%dT%H:%M:%SZ').timestamp())
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
ic = IndicatorCalculator()

# # Signals

# In[ ]:


def GenerateSignals(p, quoteVolume):    
    p_dir_span = talib.MA(p[:,0].astype(float), timeperiod=INDICATOR_MA_PERIOD)
    p_dir2nd_span = talib.MA(p[:,1].astype(float), timeperiod=INDICATOR_MA_PERIOD)

    p_dir_span_previous = p_dir_span[-2]
    p_dir2nd_span_previous = p_dir2nd_span[-2]
    
    p_dir_span = p_dir_span[-1]
    p_dir2nd_span = p_dir2nd_span[-1]

    # 2022_D1
    _entr = (p_dir_span_previous > ENTR_THR) & (p_dir_span <= ENTR_THR) & (p_dir2nd_span <= ENTR_THR2) & (quoteVolume >= VOL_LIMIT)
    _exit = (p_dir_span_previous < EXIT_THR) & (p_dir_span >= EXIT_THR)

    if _entr or _exit:
        print(p_dir_span)
        print(p_dir_span_previous)

        print(p_dir2nd_span)
        print(p_dir2nd_span_previous)

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
        cur_clean = cur.replace("-USDT","")
        fname = str(int(datetime.utcnow().timestamp())) + "_" + str(direction) + "_" + cur_clean
        f = open(SIGNAL_FOLDER + os.path.sep + fname, "w")
        f.close()
    except KeyboardInterrupt: 
        raise
    except Exception as ex:
        logging.error("Exception at writing signal file for " + str(cur))
        logging.error(ex)
        
WriteSignalFile('i-am-a-debug-signal','entry-or-exit')


# In[ ]:


def SendTelegramInfo(message):
    try:
        message = TELEGRAM_INFO_URL + datetime.utcnow().strftime("%d.%m.%Y %H:%M:%S") + '\n' + message
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

logging.info(datetime.now())

for m in suitable_markets:
    startTime = datetime.utcnow().timestamp()
    try:
        logging.info(m)
        ticks = GetTicks(m)
        tickDF = pd.DataFrame(ticks)
        tickDF.set_index("startsAt", inplace = True)
        tickDF.sort_index(inplace = True)

        # Drop latest (instable) index
        tickDF = tickDF.iloc[:-1, :]

        # Drop Quote Volume
        tickDFNoQuote = tickDF.drop("quoteVolume", axis=1)

        indDF = ic.CreateAllIndicatorsTable(tickDFNoQuote)
        normedDF = ic.NormPriceRelated(indDF)
        normedDFSlice = normedDF.tail(LOOKBACK_CNT + INDICATOR_MA_PERIOD + 1) # Plus the MA period and one before, to calc previous of MA

        # Block Generator
        try:
            xg = dg.XBlockGenerator(normedDFSlice, normedDFSlice.shape[0])        
            X = next(xg)
        except StopIteration:
            logging.warning("StopIteration occured for " + m)
            continue

        try:
            r = requests.post(PREDICTOR_REST_SERVICE_URL + "/predict", json = X.tolist())            
        except requests.exceptions.ConnectionError as cex:
            logging.error("Connection Error to predictor", cex)
            # Todo: Send error message via telegram
            break
        
        if 200 != r.status_code:
            logging.error("200 != r.status_code, it is " + str(r.status_code))
        
        content = r.content
        j_content = json.loads(content)

        if "error" in j_content:
            logging.error(j_content["error"])
            break

        # Extract predictions        
        p = np.array(j_content)

        # Get 24h data
        qv = ms.loc[m, "quoteVolume"]
        # pc = ms.loc[m, "percentChange"] / 100.0
        
        # Generate Signals
        signals = GenerateSignals(p, qv)       
        
        if signals['exit']:
            msg = m + ' exit signal'
            logging.info(msg)
            logging.info(p)
            logging.info(signals)

            SendTelegramInfo(msg)
            WriteSignalFile(m, 'exit')
        elif signals['entry']:
            msg = m + ' entry signal'
            logging.info(msg)
            logging.info(p)
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
