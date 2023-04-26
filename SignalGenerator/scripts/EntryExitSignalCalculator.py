# -*- coding: utf-8 -*-
"""
A class to calculate entry and exit signals out of a tickDF including predictions
In this case, it uses the ETF-trained "/content/bigdata/chk/FPNWithAttentionBiggerTimebased_5_FutureOnly_60days_Stable2_GPU_128LB/cp_daily_valid_06_end/model.h5" "awesome" strategy

# Todo: Clean up this file, all the variables and the dokumentation is messed up
"""

# Activate logging
import tempfile
import copy
import sys
import numpy as np
import pandas as pd
import os

import talib

import logging
logger = logging.getLogger()

if __name__ == "__main__":
    logger.setLevel(logging.INFO)


class EntryExitSignalCalculator:
    '''
    Todo add description: A class to calculate entry and exit signals out of a tickDF including predictions
    '''

    def __init__(self):
        self.ENTR_THR = 0.003
        self.ENTR_THR2 = 0.025

        self.EXIT_THR = 0.0075
        self.EXIT_THR2 = 0.01

        self.GAIN_TIMESPAN = 60
        self.INDICATOR_MINMAX_PERIOD = 28

        self.FIRST_INDICATOR_NAME = "p_future_gain"
        self.SECOND_INDICATOR_NAME = "p_future_gain_derivation"
        self.THIRD_INDICATOR_NAME = "max_past_gain"

        self.STORE_SIGNALS_DFs_FOR_DEBUGGING = True
        self.DEBUGGING_STORE_PATH = "/content/debugging"

    # For each timestep, calculate the highest possible gain within the timespan `gain_timespan`, once into the past and once into the future
    def AddMaxPastGain(self, tblIn, gain_timespan):
        __INDICATOR_NAME = "max_past_gain"

        # tblWork = copy.deepcopy(tblIn)
        tblWork = tblIn

        # Set all to 0
        tblWork.loc[:, __INDICATOR_NAME] = 0.0

        for i in range(1, tblWork.shape[0]-1):
            # Get the end index for the past
            past_index = np.max([0, i-gain_timespan])

            # Get the slice for max gain lookup from the DF
            past_slice = tblWork.iloc[past_index:i].loc[:, 'open'].values

            # Get the 'current' price
            current_price = tblWork.iloc[i].loc['open']

            if 0 == past_slice.shape[0]:
                continue

            # For the past, the min value has to be found, because in the past you wanted to buy lower
            min_past_value = np.min(past_slice)

            if 0.0 == min_past_value:
                continue

            max_past_gain = (current_price / min_past_value) - 1.0

            # Calculate the gain
            if 0.0 != min_past_value:
                tblWork.loc[tblWork.index[i], __INDICATOR_NAME] = max_past_gain

        return tblWork

    # Calculate the historical min/max values for an indicator
    def AddHistoricalMinMax(self, tblIn, featureName):
        values = tblIn.loc[:, featureName].values

        # # Shift the values to get on the historical max and not include the current value
        # values_shifted = np.empty(values.shape)
        # values_shifted[:] = np.nan
        # values_shifted[1:] = values[:-1]

        # Do not Shift the values to get on the historical max, to include the current value
        values_shifted = values

        valuesMIN = talib.MIN(values_shifted.astype(
            float), timeperiod=self.INDICATOR_MINMAX_PERIOD)
        valuesMAX = talib.MAX(values_shifted.astype(
            float), timeperiod=self.INDICATOR_MINMAX_PERIOD)

        tblIn.loc[:, featureName + '_min'] = valuesMIN
        tblIn.loc[:, featureName + '_max'] = valuesMAX

        return tblIn

    def CalcEntrySignals(self, tbl, entr_thr, entr_thr2, entr_thr3):
        # Working nicely
        _entr = (tbl.loc[:, f"{self.FIRST_INDICATOR_NAME}"] >= entr_thr) & (
            tbl.loc[:, f"{self.THIRD_INDICATOR_NAME}"] >= entr_thr2)
        tbl['entry_signal'] = _entr
        return tbl

    def CalcExitSignals(self, tbl, exit_thr, exit_thr2):
        # Working nicely
        _exit = (tbl.loc[:, f"{self.SECOND_INDICATOR_NAME}"] <= exit_thr) & (
            (tbl.loc[:, f"{self.THIRD_INDICATOR_NAME}_max"]/10.0) >= exit_thr2)
        tbl['exit_signal'] = _exit
        return tbl

    def CalculateSignals(self, tickDF: pd.DataFrame, predictions: dict):
        # Todo: Check if the required entries are in the prediction dict

        # Create a save copy
        signalsTmpDF = copy.deepcopy(tickDF)

        # Add past gain
        signalsTmpDF = self.AddMaxPastGain(signalsTmpDF, self.GAIN_TIMESPAN)

        # Crop the latest (unstable) row
        signalsTmpDF = signalsTmpDF.iloc[:-1, :]

        # Add max historical past gain
        signalsTmpDF = self.AddHistoricalMinMax(
            signalsTmpDF, self.THIRD_INDICATOR_NAME)

        # Add the predictions
        p_future_gain_array = np.zeros((signalsTmpDF.shape[0]))
        p_future_gain_array[-p['p_future'].shape[0]:] = p['p_future']
        p_future_gain_array

        p_future_gain_derivation_array = np.zeros((signalsTmpDF.shape[0]))
        p_future_gain_derivation_array[-p['p_future_derivation'].shape[0]
            :] = p['p_future_derivation']
        p_future_gain_derivation_array

        # It is working even if pandas complains
        with pd.option_context('mode.chained_assignment', None):
            signalsTmpDF.loc[:, "p_future_gain"] = p_future_gain_array
            signalsTmpDF.loc[:,
                             "p_future_gain_derivation"] = p_future_gain_derivation_array

        # Calculate the signals
        self.CalcEntrySignals(signalsTmpDF, self.ENTR_THR, self.ENTR_THR2, 0.0)
        self.CalcExitSignals(signalsTmpDF, self.EXIT_THR, self.EXIT_THR2)
