# Version 2.0 - 2022-11-16
# - Important: Order of columns in created indicator table is changed, no compatible to V1.x versions!
# - Major restructuring, code cleanup and documentation

# Version 1.3 - 2022-03-09
# Added past/future signal method and exponential smoothing of signals
# Version 1.2 - 2022-02-20
# - Added past and future max gain as y values
# Version 1.1 - 2022-02-07
# - Parameters moved to constructor
# Version 1 - 2021-11-11

# Todo: X and y generator --> a call using `for latest_blocks in xBlockGenerator:` does not work --> TypeError: 'XBlockGenerator' object is not iterable --> Change to yield?!

import numpy as np
import pandas as pd
import copy
import os
import gc
import sklearn.utils
from IndicatorCalculator import IndicatorCalculator
import talib

import logging
logger = logging.getLogger()

# Define the columns which shall be normalized batch-wise
# This is especially needed for volume-based indicators, as they have a range of around 1e7
# Todo: Pass as kwarg
COLUMNS_FOR_BATCH_NORM = ['v_AD', 'v_OBV', 'volume']
COLUMNS_FOR_BATCH_NORM_WILDCARD = ['v_ADOSC']


class XBlockGenerator:
    '''
    The `XBlockGenerator` class is used to generate time-frame slices (= X-blocks) out of a time series table of tick and indicator data.
    For this task, an input `pd.DataFrame` `tickAndIndicatorDF` is processed row by row.
    It is called 'X' because its purpose is to be used as input data for machine learning networks (== X-data).

    Every X-block is created by using a specific amount of table rows, defined by the `int` parameter `X_Block_lenght`.
    If the tick data is in hours and `X_lookback_cnt=12`, the generator would return slices of 12 hours.
    The step size between the X-blocks is 1, so the resulting DFs in the example would be: 00:00-11:00, 01:00-12:00, 02:00-13:00, ...

    Requried constructor arguments:
    - `tick_and_indicator_DF`: An `pd.DataFrame` containing a time series of tick and indicator data. This table is normally created using the `IndicatorCalculator` class.
    - `generator_batch_size`: An `int` variable defining how many X-Blocks the generator shall return on each next() call.
    - `X_Block_lenght`: This `int` variable defines how many timestamps each X-Block shall cover. Former called `X_lookback_cnt`.
    - `initial_value_norm`: A `bool` value to switch if all indicators included in `Todo` shall be normalized based on the first value in each X-block. This is then done for each indicator individually, the first value is then 0.0, all following are relative to it. Used for volume inidcators with a large spread. `True` by default.
    - `limit_volume_value`: A `bool` value to switch if the volume column shall be scaled to a maximum value of `1.0`. This can be helpful as the volume may have a large absolute value spread. `True` by default.

    Returns: A generator, X-blocks can be acquired using next()
    Raises: StopIteration if the tick and indicator table is fully consumed
    '''

    def __init__(self,
                 tick_and_indicator_DF: pd.DataFrame,
                 generator_batch_size: int,
                 X_Block_lenght: int,
                 initial_value_norm: bool = True,
                 limit_volume_value: bool = True):

        # Create a save copy of the input table
        self.tick_and_indicator_DF = copy.deepcopy(tick_and_indicator_DF)

        # At the end of the initializer, the tick and indicator values are stored as np.array instead of a pd.DataFrame to improve speed
        self.tick_and_indicator_values = None

        # -------------------------------------------------
        # Values are assigned to class-level self variables
        # -------------------------------------------------

        # How many X-Blocks the generator shall return on each next() call
        self.generator_batch_size = int(generator_batch_size)
        # The lenght of each X-Block (--> How many time-step into the past the block will reach)
        self.X_Block_lenght = int(X_Block_lenght)
        assert 1 <= self.generator_batch_size
        assert 2 <= self.X_Block_lenght

        # Attention: The value of self.slice_start_index changes during iteration, self.X_Block_lenght does not
        self.slice_start_index = X_Block_lenght

        # This variable is used in the block-gen for loop as a 'class-global' count variable, to preseve the position in the data array.
        self.block_end_index: int = 0

        # Lists of column names and indices for initial value norming (indices are the used instead of names for faster processing)
        self._col_names_initial_value_norm = None
        self._col_indices_initial_value_norm = []
        self._vol_col_index = -1

        # Flags if norming shall be done
        self.initial_value_norm = initial_value_norm
        self.limit_volume_value = limit_volume_value

        # Sort the table
        self.tick_and_indicator_DF.sort_index(inplace=True)

        # Find initial value norming columns names
        # Todo: COLUMNS_FOR_BATCH_NORM as kwarg
        self._col_names_initial_value_norm = copy.deepcopy(
            COLUMNS_FOR_BATCH_NORM)
        for c in self.tick_and_indicator_DF.columns:
            for wc in COLUMNS_FOR_BATCH_NORM_WILDCARD:
                if wc in c:
                    self._col_names_initial_value_norm.append(c)

        # Find initial value norming columns indices (based on the column names)
        for c in self._col_names_initial_value_norm:
            if c in self.tick_and_indicator_DF.columns:  # Only if present
                self._col_indices_initial_value_norm.append(
                    list(self.tick_and_indicator_DF.columns).index(c)
                )

        # Find volume column index
        if 'volume' in self.tick_and_indicator_DF.columns:
            self._vol_col_index = list(
                self.tick_and_indicator_DF.columns).index('volume')

        # Extract the values as np.array and delete the table to save memory
        self.tick_and_indicator_values = copy.deepcopy(
            self.tick_and_indicator_DF.values)
        del self.tick_and_indicator_DF
        gc.collect()

    def __next__(self):
        '''
        Create a new X data block of standard length `X_Block_lenght` and return it.

        The data type of the block is `numpy.array`.

        Raises: StopIteration if the tick and indicator table is fully consumed
        '''
        return self.__create_block__()

    def getCustomSizedSlice(self, custom_block_length: int):
        '''
        Create a new X data block of customized length `custom_block_length` and return it.
        This feature is required to fill up a batch of X-blocks in the overall process if a previous generator runs out of data.

        Optional arguments:
        - `custom_block_length`: The lenght (in time steps) of the block to be created

        The data type of the block is `numpy.array`.

        Raises: StopIteration if the tick and indicator table is fully consumed
        '''
        assert 0 < int(custom_block_length)
        return self.__create_block__(int(custom_block_length))

    def __create_block__(self, custom_block_length: int = -1):
        '''
        Create a new X data block. A customized length an be set using `custom_block_length`.
        This feature is required to fill up a batch of X-blocks in the overall process if a previous generator runs out of data.

        Requried arguments:
        - `custom_block_length`: The lenght (in time steps) of the block to be created

        The data type of the block is `numpy.array`.

        Raises: StopIteration if the tick and indicator table is fully consumed
        Todo: There are several reasons for StopIteration
        '''

        if 0 > custom_block_length:
            block_length = self.generator_batch_size
        else:
            block_length = custom_block_length

        # If the start is outside the data array's range (array is consumed), stop the iterator by raising StopIteration
        if self.slice_start_index >= self.tick_and_indicator_values.shape[0]:
            logging.debug("XBlockGenerator StopIteration - Array is consumed")
            raise StopIteration

        # Create an new empty array for the X-Block that are generated
        data_X = np.empty(
            (block_length, self.X_Block_lenght, self.tick_and_indicator_values.shape[1]))
        data_X[:] = np.nan

        # A count variable for the current position in the data_X array
        data_X_position = 0

        # Important: This for loop uses a 'class-global' count variable, as the loop is exited if the
        # desired amount of blocks has been generated, but it shall start at the next call again at self.block_end_index
        # It is increased by 1 every loop, so that every timestamp/row has the chance to be the latest in the block
        for self.block_end_index in range(self.slice_start_index, self.tick_and_indicator_values.shape[0]):

            # Pick a block-sized slice from the data array
            new_block = copy.deepcopy(
                self.tick_and_indicator_values[self.block_end_index-self.X_Block_lenght:self.block_end_index, :])

            # Norm all columns that are desired for initial value normalization
            # As it is an array, 'columns' are adressed using their indices
            if True == self.initial_value_norm:
                for ind in self._col_indices_initial_value_norm:
                    _init_val = new_block[0, ind]

                    if 0.0 != _init_val:
                        new_block[:, ind] /= _init_val
                        new_block[:, ind] -= 1.0
                    else:
                        new_block[:, ind] = 0.0

            # Norm the volume column to max == 1.0
            if True == self.limit_volume_value and 0 < self._vol_col_index:
                _vol_max = np.max(new_block[:, self._vol_col_index])
                if 0.0 < _vol_max:
                    new_block[:, self._vol_col_index] /= _vol_max

            # Place the block into the return data array
            data_X[data_X_position, :, :] = new_block
            data_X_position += 1

            # If enough block have been created, break the for loop and proceed to returning the data_X
            if data_X_position == block_length:
                # Todo: Check if block_end_index is used twice (at start of next call)
                break

        # If it is too small
        data_X = data_X[:data_X_position, :, :]

        data_X = np.nan_to_num(data_X, nan=0.0, posinf=0.0, neginf=0.0)
        # data_X = np.clip(data_X, -1000.0, 1000.0)

        self.slice_start_index = self.block_end_index + 1

        return data_X


class YDataGenerator:
    '''
    The `YDataGenerator` class is used to generate future information for training out of a time series table of tick and indicator data.
    This can for example be the assets relative price or gain in 24 hours, the direction of movement, or trade entry and exit signals.

    IMPORTANT: This class can of course not look into the future, so if you want to output the price in 24 hours, this can only be done until
    table index `len(timestamp rows) - 24`, otherwise it would really have to look into the future!

    The purpose of this class is to generate machine-learning target data (y-values) according to the X-Blocks generated by the `XBlockGenerator`.
    The X-Block contains data from the past, which is known in a live application, while the Y data is unknown in the live application, and therefore has to be predicted.
    Here, it can be generated for training purposes using historical data.

    As a price basis for calculating the y data, the `open` column of the input table `tick_DF` is used.

    Requried constructor arguments:
    - `tick_DF`: An `pd.DataFrame` containing a time series of at tick data. Only the `open` column is used.
    - `generator_batch_size`: An `int` variable defining how many y data slements the generator shall return on each next() call.
    - `first_batch_slice_start_index`: A `int` value defining the start index of the first batch. This is required to match the process of the `XBlockGenerator`, which has to start at an index more than 0 to look back.
    - `y_type_dict`: A `dict` which defines which type of y data shall be returned. As a starting point, the templates defines in this class can be used: `PARAM_DICT_TEMPLATE_Y_DATA_TYPE_DIRECTION_FLOAT`, `PARAM_DICT_TEMPLATE_Y_DATA_TYPE_DIRECTION_CATEGORICAL`, `PARAM_DICT_TEMPLATE_Y_DATA_TYPE_TRADE_SIGNALS`, `PARAM_DICT_TEMPLATE_Y_DATA_TYPE_PAST_FUTURE_GAIN`. Detailled information can be found in the README file.

    Returns: A generator, y data can be acquired using `next()`. A detailed description of the y data structure can be found in the README file. 
    Raises: StopIteration if the tick table is fully consumed
    '''

    # Todo: Add a getter for the templates to avoid changing inside a program

    # This returns the price's current direction (-> first derivation) and its changing rate (-> second derivation)
    Y_DATA_TYPE_DIRECTION_FLOAT = 0
    PARAM_DICT_TEMPLATE_Y_DATA_TYPE_DIRECTION_FLOAT = {
        "dataType": Y_DATA_TYPE_DIRECTION_FLOAT,
        # The 'direction_ma_timespan' parameter describes how fast the direction information follows price changes
        "direction_ma_timespan": 200,
        # The 'derivation_ma_timespan' parameter describes how fast the derivation information direction changes
        "derivation_ma_timespan": 100,
        # The 'direction_derivation_shift_span' parameter sets a shifting of the direction and derivation information
        "direction_derivation_shift_span": 0
    }

    # This returns the price's current direction in a categorical manner: falling, neutral, rising
    # The categories are data type `int`: falling:0, neutral:1, rising:2
    # The categories are determined by the two `float` parameters `fall_threshold` and `rise_threshold`
    Y_DATA_TYPE_DIRECTION_CATEGORICAL = 1
    PARAM_DICT_TEMPLATE_Y_DATA_TYPE_DIRECTION_CATEGORICAL = {
        "dataType": Y_DATA_TYPE_DIRECTION_CATEGORICAL,
        # The 'direction_ma_timespan' parameter describes how fast the direction information follows price changes
        "direction_ma_timespan": 200,
        # The 'derivation_ma_timespan' parameter describes how fast the derivation information direction changes
        "derivation_ma_timespan": 100,
        # The 'direction_derivation_shift_span' parameter sets a shifting of the direction and derivation information
        "direction_derivation_shift_span": 0,
        "fall_threshold": -0.2,
        "rise_threshold": 0.1
    }

    # Creation of trade signals (entry/exit points) out of direction and direction derivation information
    # Entry signals are controlled by 3 parameters: `entr_thr1`, `entr_thr2`, `entr_thr3` -> Todo: Describe in detail
    # Exit signals are controlled by 2 parameters: `exit_thr1`, `exit_thr2` -> Todo: Describe in detail
    Y_DATA_TYPE_TRADE_SIGNALS = 2
    PARAM_DICT_TEMPLATE_Y_DATA_TYPE_TRADE_SIGNALS = {
        "dataType": Y_DATA_TYPE_TRADE_SIGNALS,
        "direction_ma_timespan": 200,
        "derivation_ma_timespan": 100,
        "direction_derivation_shift_span": 0,
        "future_direction_shift_timespan": 24,
        "entr_thr1": 0.9,
        "entr_thr2": 0.8,
        "entr_thr3": 0.0,
        "exit_thr1": -0.5,
        "exit_thr2": 0.1
    }

    # This returns two maximum possible gains: One that could have been achieved in the last `gain_timespan` ticks,
    # and the one that is possible in the next `gain_timespan` ticks if you would buy now.
    # Usage: Buy if there has not been past gain but if there will be future gain. Sell the other way round.
    Y_DATA_TYPE_PAST_FUTURE_GAIN = 3
    PARAM_DICT_TEMPLATE_Y_DATA_TYPE_PAST_FUTURE_GAIN = {
        "dataType": Y_DATA_TYPE_PAST_FUTURE_GAIN,
        "gain_timespan": 12,
        "direction_ma_timespan": 48,
        "derivation_ma_timespan": 48
    }

    # Todo: Add y data type none to not init a generator

    def __init__(self,
                 tick_DF: pd.DataFrame,
                 generator_batch_size: int,
                 first_batch_slice_start_index: int,
                 y_type_dict: dict):
        # Todo doku: y data is calculated right at class init

        # Store the y type dict
        self.y_type_dict = y_type_dict

        # Create a save copy of the open column
        self.yDataDF = copy.deepcopy(pd.DataFrame(tick_DF.loc[:, 'open']))

        self.slice_size = generator_batch_size
        self.batch_slice_start_index = first_batch_slice_start_index

        self.i = 0

        # Sort the table by timestamp
        self.yDataDF.sort_index(inplace=True)

        # Data Type Y_DATA_TYPE_DIRECTION_FLOAT or Y_DATA_TYPE_DIRECTION_CATEGORICAL or Y_DATA_TYPE_TRADE_SIGNALS
        if self.Y_DATA_TYPE_DIRECTION_FLOAT == self.y_type_dict["dataType"] or self.Y_DATA_TYPE_DIRECTION_CATEGORICAL == self.y_type_dict["dataType"] or self.Y_DATA_TYPE_TRADE_SIGNALS == self.y_type_dict["dataType"]:
            # Check if all necessary values are present in the y-type descriptor dict
            necessary_values = [
                "direction_ma_timespan", "derivation_ma_timespan", "direction_derivation_shift_span"]
            for nev in necessary_values:
                if not nev in self.y_type_dict.keys():
                    raise KeyError(
                        f"'{nev}' has to be defined in y-type descriptor dict")

            # Get the values from the dict
            direction_ma_timespan = int(
                self.y_type_dict["direction_ma_timespan"])
            derivation_ma_timespan = int(
                self.y_type_dict["derivation_ma_timespan"])
            direction_derivation_shift_span = int(
                self.y_type_dict["direction_derivation_shift_span"])

            assert 0 < direction_ma_timespan
            assert 0 < derivation_ma_timespan

            # Calculate moving direction of price and its change rate
            # This is done by calculating a shifted first and second derivation
            # The return of the method is: shifted_ma, derivation_first, derivation_first_shifted_ma, derivation_second
            # Only the 3. and 4. element are used, which are the smoothed first derivation and the 'normal' second one
            _, _, direction, directionDerivation = self.calculateShiftedDerivations(
                self.yDataDF.values.flatten(),
                direction_ma_timespan,
                derivation_ma_timespan)  # Todo : np.squeeze instead of flatten?

            # All nan values shall simply be 0
            # Todo: Add a switch to cut them out, as they are wrong value, which may disturb training
            direction = np.nan_to_num(direction, nan=0)
            directionDerivation = np.nan_to_num(directionDerivation, nan=0)

            # Smooth out the direction derivation
            # Todo: Add a switch
            directionDerivation, _, _, _ = self.calculateShiftedDerivations(
                directionDerivation,
                derivation_ma_timespan,
                1)

            # All nan values shall simply be 0
            # Todo: Add a switch to cut them out, as they are wrong value, which may disturb training
            directionDerivation = np.nan_to_num(directionDerivation, nan=0)

            # Class-wide storage of direction data
            # It is possible to shift the direction and the directionDerivation array to acquire value from the future (e.g. the dir/derv in 24 hours)
            if 0 < direction_derivation_shift_span:
                logging.debug(
                    f"it has been shifted by {direction_derivation_shift_span}")
                self.direction = np.zeros(direction.shape)
                self.direction[:-direction_derivation_shift_span] = direction[direction_derivation_shift_span:]
                self.direction = np.nan_to_num(self.direction, nan=0)

                self.directionDerivation = np.zeros(directionDerivation.shape)
                self.directionDerivation[:-
                                         direction_derivation_shift_span] = directionDerivation[direction_derivation_shift_span:]
                self.directionDerivation = np.nan_to_num(
                    self.directionDerivation, nan=0)
            else:
                self.direction = direction
                self.directionDerivation = directionDerivation

            # Limit the value range to avoid excess of the derivations
            self.direction = np.tanh(self.direction * 1000.0)
            self.directionDerivation = np.tanh(self.directionDerivation)

            # If the data type is Y_DATA_TYPE_DIRECTION_CATEGORICAL, further processing into categories is required
            if self.Y_DATA_TYPE_DIRECTION_CATEGORICAL == self.y_type_dict["dataType"]:
                # Check if all necessary values are present in the y-type descriptor dict
                necessary_values = [
                    "fall_threshold", "rise_threshold"]
                for nev in necessary_values:
                    if not nev in self.y_type_dict.keys():
                        raise KeyError(
                            f"'{nev}' has to be defined in y-type descriptor dict")

                fall_threshold = float(self.y_type_dict["fall_threshold"])
                rise_threshold = float(self.y_type_dict["rise_threshold"])

                # Calculate falling category
                self.directionFalling = self.direction < fall_threshold

                # Rise can only occur at positive dir derivation and if it is not falling at this moment
                self.directionRising = (self.direction > rise_threshold) & (
                    self.directionDerivation > 0) & (False == self.directionFalling)

                # Convert into numerical
                self.directionCategory = np.empty(self.direction.shape)
                self.directionCategory[:] = 1  # Neutral category

                self.directionCategory = self.directionCategory + \
                    (self.directionFalling * -1) + (self.directionRising * 1)
                self.directionCategory = self.directionCategory.astype(int)

            # If the data type is Y_DATA_TYPE_DIRECTION_CATEGORICAL, further processing into trade signals is required
            if self.Y_DATA_TYPE_TRADE_SIGNALS == self.y_type_dict["dataType"]:
                # Check if all necessary values are present in the y-type descriptor dict
                necessary_values = [
                    "future_direction_shift_timespan",
                    "entr_thr1",
                    "entr_thr2",
                    "entr_thr3",
                    "exit_thr1",
                    "exit_thr2"
                ]
                for nev in necessary_values:
                    if not nev in self.y_type_dict.keys():
                        raise KeyError(
                            f"'{nev}' has to be defined in y-type descriptor dict")

                # Get the values from the dict
                future_direction_shift_timespan = int(
                    self.y_type_dict["future_direction_shift_timespan"])
                entr_thr1 = float(self.y_type_dict["entr_thr1"])
                entr_thr2 = float(self.y_type_dict["entr_thr2"])
                entr_thr3 = float(self.y_type_dict["entr_thr3"])
                exit_thr1 = float(self.y_type_dict["exit_thr1"])
                exit_thr2 = float(self.y_type_dict["exit_thr2"])

                # Shift the direction array to acquire future direction information for trading signals (to answer: will it be profitable in the future?)
                _direction_futureshifted = np.empty(self.direction.shape)
                _direction_futureshifted[:] = 0.0
                _direction_futureshifted[:-
                                         future_direction_shift_timespan] = self.direction[future_direction_shift_timespan:]

                # Calculate entry and exit signals based on three entry and two exit parameters
                self.entry = (self.direction >= entr_thr1) & (
                    _direction_futureshifted >= entr_thr2) & (self.directionDerivation >= entr_thr3)
                self.exit = (self.direction <= exit_thr1) & (
                    _direction_futureshifted <= exit_thr2)

        # Data type Y_DATA_TYPE_PAST_FUTURE_GAIN
        if self.Y_DATA_TYPE_PAST_FUTURE_GAIN == self.y_type_dict["dataType"]:
            # Check if all necessary values are present in the y-type descriptor dict
            necessary_values = [
                "gain_timespan", "direction_ma_timespan", "derivation_ma_timespan"]
            for nev in necessary_values:
                if not nev in self.y_type_dict.keys():
                    raise KeyError(
                        f"'{nev}' has to be defined in y-type descriptor dict")

            # Get the values from the dict
            gain_timespan = float(self.y_type_dict["gain_timespan"])
            direction_ma_timespan = int(
                self.y_type_dict["direction_ma_timespan"])
            derivation_ma_timespan = int(
                self.y_type_dict["derivation_ma_timespan"])

            # Initialize new columns in the DataFrame for past and future gain
            self.yDataDF['max_past_gain'] = 0.0
            self.yDataDF['max_future_gain'] = 0.0

            # For each timestep, calculate the highest possible gain within the timespan `gain_timespan`, once into the past and once into the future
            for i in range(1, self.yDataDF.shape[0]-1):
                # Get the end indices for past and future
                past_index = np.max([0, i-gain_timespan])
                future_index = np.min(
                    [self.yDataDF.shape[0], i+gain_timespan])

                # Get the slices for max gain lookup from the DF
                past_slice = self.yDataDF.iloc[past_index:i].loc[:,
                                                                 'open'].values
                future_slice = self.yDataDF.iloc[i +
                                                 1:future_index].loc[:, 'open'].values

                # Get the 'current' price
                current_price = self.yDataDF.iloc[i].values[0]

                if 0 == past_slice.shape[0] or 0 == future_slice.shape[0]:
                    continue

                # For the past, the min value has to be found, because in the past you wanted to buy lower
                # For the future, the max value has to be found, because in the future you wanted to sell higher
                min_past_value = np.min(past_slice)
                max_future_value = np.max(future_slice)

                # Calculate the gains
                if 0.0 != min_past_value:
                    self.yDataDF.loc[self.yDataDF.index[i], 'max_past_gain'] = (
                        current_price / min_past_value) - 1.0

                if 0.0 != current_price:
                    self.yDataDF.loc[self.yDataDF.index[i], 'max_future_gain'] = (
                        max_future_value / current_price) - 1.0

            # Convert the table-data into arrays for futher processing
            max_past_gain = self.yDataDF.loc[:, 'max_past_gain'].values
            max_future_gain = self.yDataDF.loc[:,
                                               'max_future_gain'].values

            # Smooth the gain out and calculate its derivation
            self.max_past_gain_ma, _, self.max_past_gain_dir, _ = self.calculateShiftedDerivations(
                max_past_gain,
                direction_ma_timespan,
                derivation_ma_timespan)
            self.max_future_gain_ma, _, self.max_future_gain_dir, _ = self.calculateShiftedDerivations(
                max_future_gain,
                direction_ma_timespan,
                derivation_ma_timespan)

            # Replace all nan values just by zero
            self.max_past_gain_ma = np.nan_to_num(self.max_past_gain_ma, nan=0)
            self.max_future_gain_ma = np.nan_to_num(
                self.max_future_gain_ma, nan=0)
            self.max_past_gain_dir = np.nan_to_num(
                self.max_past_gain_dir, nan=0)
            self.max_future_gain_dir = np.nan_to_num(
                self.max_future_gain_dir, nan=0)

        # Delete the yDataDF DataFrame to save memory
        del self.yDataDF
        gc.collect()

    def __next__(self):
        return self.__create_block__()

    def getCustomSizedSlice(self, custom_slice_size):
        assert 0 < custom_slice_size
        return self.__create_block__(custom_slice_size)

    def calculateShiftedDerivations(self, input_array: np.array, ma_timespan: int, derivation_ma_timespan: int = 0):
        '''
        This method is used to calulate the first and second derivation of the price data.
        A speciality of the method is that the 'center' of the derivation calculation is at the 'current' timestamp, so that derivation takes values from the past and the future into account.

        To avoid excess values, the price data is smoothed before derivation calulation.

        As it is allowed to 'look into the future' in the y data generation, the mean value is shifted by `ma_timespan / 2`, to provide meaning as well into the past as into the future.
        On this data, the derivation calculation is then applied.

        Derivation data rows on the end of the table are set to `np.nan`, as no future data is available.

        Requried arguments:
        - `input_array`: An `np.array` containing one time series of data.
        - `ma_timespan`: An `int` variable defining the timespan over which a moving average is apllied on the input data before calculating the first derivation.
        - `derivation_ma_timespan`: An `int` variable defining the timespan over which a moving average is apllied on the first derivation data before calculating the second derivation.

        Returns:
        A `tuple` of `np.array` containing 4 elements:
        - `shifted_ma`: A moving average of the input data, shifted to its center
        - `derivation_first`: The first derivation of `shifted_ma`. Attention: It is calculated as percent value (now-prev)/prev.
        - `derivation_first_shifted_ma`: A moving average of the first derivation data, shifted to its center
        - `derivation_second`: The second derivation of `shifted_ma`, or more precise, the first derivation of `derivation_first_shifted_ma`. Attention: It is calculated as percent value (now-prev)/prev.
        '''

        # Calculate the moving average of the input data
        ma = talib.MA(input_array.astype(float), timeperiod=ma_timespan)

        # Move the MA data forward half the span, so that its period center is 'now'
        # This can only be done if the ma_timespan is bigger than 1, otherwise there is no middle
        # Elements non-available are filled with np.nan
        shifted_ma = np.empty(ma.shape)
        shifted_ma[:] = np.nan
        if 1 < ma_timespan:
            shifted_ma[:-int(ma_timespan/2)] = ma[int(ma_timespan/2):]
        else:
            shifted_ma = ma

        # If the shifted_ma array contains only nan (data too short), all outputs as nan
        if all(np.isnan(shifted_ma)):
            return shifted_ma, shifted_ma, shifted_ma, shifted_ma

        # Calculate the first deviation
        # Attention: It is calculated as percent value (now-prev)/prev
        derivation_first = talib.ROCP(shifted_ma, timeperiod=1)

        if 0 == derivation_ma_timespan:
            # This is the exit scenerio for recursion
            derivation_first_shifted_ma, derivation_second = np.zeros(
                derivation_first.shape), np.zeros(derivation_first.shape)
        else:
            # Calculate the second derivation
            derivation_first_shifted_ma, derivation_second, _, _ = self.calculateShiftedDerivations(
                derivation_first, derivation_ma_timespan)

        return shifted_ma, derivation_first, derivation_first_shifted_ma, derivation_second

    def __create_data_slice__(self, dataArray: np.array, startIndex: int, endIndex: int):
        '''
        Method for extracting a time-slice out the whole y data array

        Requried arguments:
        - `input_array`: An `np.array` containing the whole y data
        - `startIndex`: An `int` defining the start index of the slice
        - `endIndex`: An `int` defining the end index of the slice

        Returns:
        A `np.array` containing the data slice.
        '''

        return dataArray[startIndex:endIndex]

    def __create_block__(self, custom_slice_size=None):
        if custom_slice_size is None:
            _local_slice_size = self.slice_size
        else:
            _local_slice_size = custom_slice_size

        # Todo: Depending on Y type, not working for 2!
        # Raise StopIteration if table is consumed
        # if self.slice_start_index >= self.direction.shape[0]:
        #     logging.info(
        #         "Stop Iteration in Line 143 - Table consumed in y gen")
        #     raise StopIteration

        # Prepare a return array, it is filled in the if statements depending on y type
        returnArray = None

        # Return statements for different y data types
        if self.Y_DATA_TYPE_DIRECTION_FLOAT == self.y_type_dict["dataType"]:
            direction_slice = self.__create_data_slice__(self.direction,
                                                         self.batch_slice_start_index,
                                                         min([
                                                             self.direction.shape[0], self.batch_slice_start_index +
                                                             _local_slice_size
                                                         ])
                                                         )

            directionDerivation_slice = self.__create_data_slice__(self.directionDerivation,
                                                                   self.batch_slice_start_index,
                                                                   min([
                                                                       self.direction.shape[0], self.batch_slice_start_index +
                                                                       _local_slice_size
                                                                   ])
                                                                   )

            # The `direction_float_return_array` contains both directions and derivation
            direction_float_return_array = np.empty(
                (direction_slice.shape[0], 2))
            direction_float_return_array[:, 0] = direction_slice
            direction_float_return_array[:, 1] = directionDerivation_slice

            returnArray = direction_float_return_array

        elif self.Y_DATA_TYPE_DIRECTION_CATEGORICAL == self.y_type_dict["dataType"]:
            category_slice = self.__create_data_slice__(self.directionCategory,
                                                        self.batch_slice_start_index,
                                                        min([
                                                            self.directionCategory.shape[0], self.batch_slice_start_index +
                                                            _local_slice_size
                                                        ])
                                                        )

            returnArray = category_slice

        elif self.Y_DATA_TYPE_PAST_FUTURE_GAIN == self.y_type_dict["dataType"]:

            # Create slices
            max_past_gain_ma_slice = self.__create_data_slice__(self.max_past_gain_ma,
                                                                self.batch_slice_start_index,
                                                                min([
                                                                    self.max_past_gain_ma.shape[0], self.batch_slice_start_index +
                                                                    _local_slice_size
                                                                ])
                                                                )

            max_future_gain_ma_slice = self.__create_data_slice__(self.max_future_gain_ma,
                                                                  self.batch_slice_start_index,
                                                                  min([
                                                                      self.max_future_gain_ma.shape[0], self.batch_slice_start_index +
                                                                      _local_slice_size
                                                                  ])
                                                                  )

            max_past_gain_dir_slice = self.__create_data_slice__(self.max_past_gain_dir,
                                                                 self.batch_slice_start_index,
                                                                 min([
                                                                     self.max_past_gain_dir.shape[0], self.batch_slice_start_index +
                                                                     _local_slice_size
                                                                 ])
                                                                 )

            max_future_gain_dir_slice = self.__create_data_slice__(self.max_future_gain_dir,
                                                                   self.batch_slice_start_index,
                                                                   min([
                                                                       self.max_future_gain_dir.shape[0], self.batch_slice_start_index +
                                                                       _local_slice_size
                                                                   ])
                                                                   )

            gains = np.empty((max_past_gain_ma_slice.shape[0], 4))

            # Gain values
            gains[:, 0] = np.tanh(max_past_gain_ma_slice)
            gains[:, 1] = np.tanh(max_future_gain_ma_slice)

            # Gain derivation values
            gains[:, 2] = np.tanh(max_past_gain_dir_slice)
            gains[:, 3] = np.tanh(max_future_gain_dir_slice)

            returnArray = gains

        elif self.Y_DATA_TYPE_TRADE_SIGNALS == self.y_type_dict["dataType"]:
            entry_slice = self.__create_data_slice__(self.entry,
                                                     self.batch_slice_start_index,
                                                     min([
                                                         self.entry.shape[0], self.batch_slice_start_index +
                                                         _local_slice_size
                                                     ])
                                                     )

            exit_slice = self.__create_data_slice__(self.exit,
                                                    self.batch_slice_start_index,
                                                    min([
                                                        self.exit.shape[0], self.batch_slice_start_index +
                                                        _local_slice_size
                                                    ])
                                                    )

            # Placeholder for neutral (-> No entry and no exit signal) to allow use of categorical crossentropy loss on ML training
            neutral_slice = np.logical_not(
                np.logical_or(entry_slice, exit_slice))

            signals = np.empty((entry_slice.shape[0], 3))
            signals[:, 0] = entry_slice
            signals[:, 1] = exit_slice
            signals[:, 2] = neutral_slice

            returnArray = signals.astype(int)

        # If another or no data type is specified (e.g. in live-predicing signals), just return an zeros array for compatibility reasons
        else:
            returnArray = np.zeros((_local_slice_size))

        # Increase the batch slice start index for the next call
        self.batch_slice_start_index += _local_slice_size

        return returnArray


class FileListToDataStream:
    '''
    The `FileListToDataStream` class is used to generate a stream of `X-Blocks` and their according `y-data` out of a list of OHLCV (Open/High/Low/Close/Volume) .csv files.
    In short, `X-Blocks` are a slice of data from the past, where `y-data` contains information about future gain / trade signals. For detailled
    information, please take a look at the description of the classes `XBlockGenerator` and `YDataGenerator`.



    Requried constructor arguments:
    - `fileList`: A `list` of `string` with complete paths to the tick data (OHLCV) files.
    - `todo`: some stuff


    Returns: Todo: Two  generators ??
    Raises: Todo: ?? StopIteration if the tick table is fully consumed
    '''

    def __init__(self, fileList, batch_size,
                 base_path, smooth_cnt, smooth_cnt2,
                 X_Block_lenght, y_lookahead_cnt, gain_lookaround_cnt, y_type, parallel_generators=4,
                 shuffle=True, random_seed=42, rise_gain_threshold=0.0, fall_gain_threshold=0.0, shortspan=6, midspan=48, longspan=120, y_exponent=1.0,
                 expected_gain_lookforward=48, entr_thr=0.0, entr_thr2=0.0, entr_thr3=0.0, exit_thr=0.0, exit_thr2=0.0,
                 norm_price_related_indicators=True, verbose=False, initial_value_norm=True, limit_volume_value=True,
                 **kwargs
                 ):

        # Default values
        # Todo: Write doku for kwargs
        self.shortspan = 6
        self.midspan = 48
        self.longspan = 120
        self.random_seed = 42
        self.shuffle = False
        self.verbose = False
        self.initial_value_norm = True
        self.limit_volume_value = True
        self.norm_price_related_indicators = True

        # Parse kwargs

        # Indicator time spans
        if "shortspan" in kwargs.keys():
            self.shortspan = int(kwargs["shortspan"])
            assert 2 <= self.shortspan
        if "midspan" in kwargs.keys():
            self.shortspan = int(kwargs["midspan"])
            assert 2 <= self.midspan
            assert self.shortspan < self.midspan
        if "longspan" in kwargs.keys():
            self.longspan = int(kwargs["longspan"])
            assert 2 <= self.longspan
            assert self.midspan < self.longspan

        # Stuff
        if "random_seed" in kwargs.keys():
            self.random_seed = int(kwargs["random_seed"])
        if "shuffle" in kwargs.keys():
            if True == kwargs["shuffle"]:
                self.shuffle = True
        if "verbose" in kwargs.keys():
            if True == kwargs["verbose"]:
                self.verbose = True
        if "initial_value_norm" in kwargs.keys():
            if False == kwargs["initial_value_norm"]:
                self.initial_value_norm = False
        if "limit_volume_value" in kwargs.keys():
            if False == kwargs["limit_volume_value"]:
                self.limit_volume_value = False
        if "norm_price_related_indicators" in kwargs.keys():
            if False == kwargs["norm_price_related_indicators"]:
                self.norm_price_related_indicators = False

        assert 0 < parallel_generators
        assert 0 < batch_size

        # Check if the batch size suits the number of parallel X/y generators, as each generator has to provide a integer count of elements
        if 0 != batch_size % parallel_generators:
            suitable_batch_size = int(
                np.floor(batch_size / parallel_generators) * parallel_generators)

            raise ValueError(
                f"Non-Suitable batch size {batch_size} for {parallel_generators} parallel generators. You could use a batch size of {suitable_batch_size}.")

        self.batch_size = batch_size
        # Todo important: Check if file list len is at least parallel_generators
        self.parallel_generators = parallel_generators

        # Each X/y generator shall provide the same amount of data to the overall batch
        self.batch_size_generator = int(batch_size / parallel_generators)

        self.y_lookahead_cnt = y_lookahead_cnt
        self.X_Block_lenght = X_Block_lenght
        self.gain_lookaround_cnt = gain_lookaround_cnt

        # self.expected_gain_lookforward = expected_gain_lookforward # Todo: Move to y descr dict
        # self.entr_thr = entr_thr  # Todo: Move to y descr dict
        # self.entr_thr2 = entr_thr2  # Todo: Move to y descr dict
        # self.entr_thr3 = entr_thr3  # Todo: Move to y descr dict
        # self.exit_thr = exit_thr  # Todo: Move to y descr dict
        # self.exit_thr2 = exit_thr2  # Todo: Move to y descr dict

        # 0 for categorical, 1 for float, 2 for gain lookaround, 3 for entry / exit
        # Todo: Remove this var and include into dict
        self.y_type = y_type

        # A storage for the X and y data generators, there will be `parallel_generators` instances of them
        self.X_generators = []
        self.y_generators = []

        # Initiate an IndicatorCalculator instance
        self.indicatorCalculator = IndicatorCalculator(
            self.shortspan, self.midspan, self.longspan, verbose=self.verbose)

        # Create a modification-safe file list and shuffle them if necessary
        if True == self.shuffle:
            self.fileList = copy.deepcopy(sklearn.utils.shuffle(
                fileList, random_state=self.random_seed))
        else:
            self.fileList = copy.deepcopy(fileList)

        # Initalize all the parallel X and y generators. Each generator loads a new file from the file list
        for i in range(parallel_generators):
            # Get a file name
            filePath = self.fileList.pop()

            XGen, yGen = self.__initGenerators__(
                filePath, self.batch_size_generator)

            self.X_generators.append(XGen)
            self.y_generators.append(yGen)

            # Todo: Logging only under verbose?!
            fileName = filePath.split(os.pathsep)[0]
            logging.info(
                f"File '{fileName}' loaded, {len(self.fileList)} left")

    def __initGenerators__(self, filePath):
        # Todo: Pass a list of full paths to class
        # _fullname = os.path.join(self.base_path, fn)

        tickDF = pd.read_csv(filePath, encoding="utf-8",
                             header=0, index_col='startsAt')

        # Todo important: Do a check if OHLC(V) columns are present and no more or less

        # _tickDF.dropna(inplace=True)  # Does not work with cmc info being nan

        # If there is a quote volume, drop it
        # if 'quoteVolume' in tickDF.columns:
        #     tickDF.drop('quoteVolume', axis=1, inplace=True)

        # Todo: If no V column then exclude these indicators
        tickAndIndicatorDF = self.indicatorCalculator.CreateAllIndicatorsTable(
            tickDF)

        # If price related norming is necessary, do it, otherwise just copy the tickAndIndicatorDF table
        if True == self.norm_price_related_indicators:
            normedDF = self.indicatorCalculator.NormPriceRelated(
                tickAndIndicatorDF)
        else:
            normedDF = tickAndIndicatorDF

        # Init the X block generator
        XGen = XBlockGenerator(
            normedDF,
            self.batch_size_generator,
            self.X_Block_lenght,
            self.initial_value_norm,
            self.limit_volume_value
        )

        # Init the y data generator
        yGen = YDataGenerator(
            normedDF,
            self.batch_size_generator,
            self.X_Block_lenght,
            self.y_type
        )

        return XGen, yGen

    def __next__(self):
        _shape_invalid = True

        while _shape_invalid:
            _X_data = None
            _y_data = None
            for i in range(self.parallel_generators):
                try:
                    _X = next(self.X_generators[i])
                    _y = next(self.y_generators[i])[self.y_type]
                except StopIteration:  # If the generator stop internally
                    # Check if the file list is empty
                    if 0 == len(self.fileList):
                        logging.info(
                            "Stop Iteration in Line 256 - 0 == len(self.fileList)")
                        raise StopIteration

                    for retry in range(10+1):
                        _fn = self.fileList.pop()
                        logging.info("File " + str(_fn) + " loaded")
                        logging.info("Files left: " + str(len(self.fileList)))
                        self.X_generators[i], self.y_generators[i] = self.__initGenerators__(
                            _fn, self.batch_size_generator)
                        try:
                            _X = next(self.X_generators[i])
                            _y = next(self.y_generators[i])[self.y_type]

                            break
                        except StopIteration:
                            logging.warning(
                                "Stop Iteration in Line 267 on getting new generators, retry " + str(retry))
                            if 10 == retry:
                                raise Exception(
                                    "10 == retry on getting new generators")

                _fn = "I am inited"
                # Check if the generator is fully consumed
                if _X.shape[0] != self.batch_size_generator:
                    # Check if the file list is empty
                    if 0 == len(self.fileList):
                        logging.info(
                            "Stop Iteration in Line 269 - 0 == len(self.fileList)")
                        raise StopIteration

                    _fn = self.fileList.pop()
                    # Todo: Full file names: Split by folder seperator and show only the last element
                    logging.info("File " + str(_fn) + " loaded")
                    logging.info("Files left: " + str(len(self.fileList)))
                    self.X_generators[i], self.y_generators[i] = self.__initGenerators__(
                        _fn, self.batch_size_generator)

                    # Fill up the missing elements from the new file
                    try:
                        _missing_cnt = self.batch_size_generator - _X.shape[0]
                        _missing_X = self.X_generators[i].getCustomSizedSlice(
                            _missing_cnt)

                        # Todo: Check if this works
                        # _missing_y = self.y_generators[i].getCustomSizedSlice(_missing_cnt)[
                        #     self.y_type]
                        _missing_y = self.y_generators[i].getCustomSizedSlice(
                            _missing_cnt)

                        _X = np.concatenate((_X, _missing_X))
                        _y = np.concatenate((_y, _missing_y))
                    except StopIteration:
                        logging.warning(
                            "Caught StopIteration in filling missing values for " + str(_fn))
                        # Check if the file list is empty
                        if 0 == len(self.fileList):
                            logging.info(
                                "Stop Iteration in Line 295 - 0 == len(self.fileList)")
                            raise StopIteration

                        _fn = self.fileList.pop()
                        logging.info("File " + str(_fn) + " loaded")
                        logging.info("Files left: " + str(len(self.fileList)))
                        self.X_generators[i], self.y_generators[i] = self.__initGenerators__(
                            _fn, self.batch_size_generator)

                        _X = next(self.X_generators[i])

                        # Todo: Check if this works
                        # _y = next(self.y_generators[i])[self.y_type]
                        _y = next(self.y_generators[i])

                #
                # if 0 == self.y_type:
                #     assert 1 == len(_y.shape)
                # elif 1 == self.y_type or 2 == self.y_type or 3 == self.y_type:
                #     assert 2 == len(_y.shape)

                # Todo: Workaround to fix if shape is not (xx,1), finally check why this happens
                # if 2 != len(_y.shape):
                #  _y = _y.reshape((-1,1))

                # Ensure that the time-dimension has the same size
                assert _X.shape[0] == _y.shape[0]

                if _X_data is None:
                    _X_data = _X
                    _y_data = _y
                else:
                    _X_data = np.concatenate((_X_data, _X))
                    _y_data = np.concatenate((_y_data, _y))

                _y_data = _y_data

            # Try again if size is wrong
            if self.batch_size != _X_data.shape[0]:
                logging.warning("\nself.batch_size != _X_data.shape[0]")
                logging.warning("self.batch_size: " + str(self.batch_size))
                logging.warning("_X_data.shape[0]: " + str(_X_data.shape[0]))
                logging.warning("_X_data.shape: " + str(_X_data.shape))
                continue
            else:
                _shape_invalid = False

            if self.shuffle:
                _X_data, _y_data = sklearn.utils.shuffle(
                    _X_data, _y_data, random_state=self.random_seed)

        # End if everything is consumed
        if self.batch_size != _X_data.shape[0]:
            logging.info(
                "Stop Iteration in Line 376 - self.batch_size != _X_data.shape[0]")
            logging.info("self.batch_size: " + str(self.batch_size))
            logging.info("_X_data.shape[0]: " + str(_X_data.shape[0]))
            logging.info("_X_data.shape: " + str(_X_data.shape))
            raise StopIteration

        return _X_data, _y_data
