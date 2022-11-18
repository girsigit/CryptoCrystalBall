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

import numpy as np
import pandas as pd
import copy
import os
import gc
import sklearn.utils
from IndicatorCalculator import IndicatorCalculator
from sklearn.preprocessing import OneHotEncoder
import talib

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
                 generator_batch_size,
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

        self.generator_batch_size = int(generator_batch_size)  # How many X-Blocks the generator shall return on each next() call
        self.X_Block_lenght = int(X_Block_lenght) # The lenght of each X-Block (--> How many time-step into the past the block will reach)
        assert 1 <= self.generator_batch_size
        assert 2 <= self.X_Block_lenght

        # Attention: The value of self.slice_start_cnt changes during iteration, self.X_Block_lenght does not
        self.slice_start_cnt = X_Block_lenght

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
            if c in self.tick_and_indicator_DF.columns: # Only if present
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
        if self.slice_start_cnt >= self.tick_and_indicator_values.shape[0]:
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
        for self.block_end_index in range(self.slice_start_cnt, self.tick_and_indicator_values.shape[0]):

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

        self.slice_start_cnt = self.block_end_index + 1

        return data_X


class YGainGenerator:
    def __init__(self, inDF, slice_size,
                 rise_threshold, fall_threshold,
                 smooth_cnt, smooth_cnt2,
                 lookback_cnt, y_lookahead_cnt, gain_lookaround_cnt,
                 expected_gain_lookforward, entr_thr, entr_thr2, entr_thr3,
                 exit_thr, exit_thr2):
        self.gainDF = copy.deepcopy(pd.DataFrame(inDF.loc[:, 'open']))
        self.slice_size = slice_size
        self.slice_start_cnt = lookback_cnt
        self.rise_threshold = rise_threshold
        self.fall_threshold = fall_threshold
        self.gain_lookaround_cnt = gain_lookaround_cnt
        self.expected_gain_lookforward = expected_gain_lookforward
        self.entr_thr = entr_thr
        self.entr_thr2 = entr_thr2
        self.entr_thr3 = entr_thr3
        self.exit_thr = exit_thr
        self.exit_thr2 = exit_thr2
        self.i = 0

        # Sort the table
        self.gainDF.sort_index(inplace=True)

        # Calculate direction
        _, _, _direction, _directiondev2nd = self.smoothMAandROC(
            copy.deepcopy(self.gainDF.values).flatten(), smooth_cnt, smooth_cnt2)
        _direction = np.nan_to_num(_direction, nan=0)
        _directiondev2nd = np.nan_to_num(_directiondev2nd, nan=0)

        # Re-Smooth 2nd
        _directiondev2nd, _, _, _ = self.smoothMAandROC(
            _directiondev2nd, smooth_cnt2, 1)
        _directiondev2nd = np.nan_to_num(_directiondev2nd, nan=0)

        # Calculate Lookaround
        self.gainDF['max_past_gain'] = 0.0
        self.gainDF['max_future_gain'] = 0.0

        # If gain_lookaround_cnt is set to nan, skip the calculation (used in prediction)
        if not np.isnan(self.gain_lookaround_cnt):
            for i in range(1, self.gainDF.shape[0]-1):
                past_index = np.max([0, i-lookback_cnt])
                future_index = np.min([self.gainDF.shape[0], i+lookback_cnt])

                past_slice = self.gainDF.iloc[past_index:i].loc[:,
                                                                'open'].values
                future_slice = self.gainDF.iloc[i +
                                                1:future_index].loc[:, 'open'].values

                current_value = self.gainDF.iloc[i].values[0]

                if 0 == past_slice.shape[0] or 0 == future_slice.shape[0]:
                    continue

                min_past_value = np.min(past_slice)
                max_future_value = np.max(future_slice)

                if 0.0 != min_past_value:
                    self.gainDF.loc[self.gainDF.index[i], 'max_past_gain'] = (
                        current_value / min_past_value) - 1.0

                if 0.0 != current_value:
                    self.gainDF.loc[self.gainDF.index[i], 'max_future_gain'] = (
                        max_future_value / current_value) - 1.0

        self.max_past_gain = self.gainDF.loc[:, 'max_past_gain'].values
        self.max_future_gain = self.gainDF.loc[:, 'max_future_gain'].values

        self.max_past_gain_ma, self.max_past_gain_dir, _, _ = self.smoothMAandROC(
            self.max_past_gain, smooth_cnt, smooth_cnt2)
        self.max_future_gain_ma, self.max_future_gain_dir, _, _ = self.smoothMAandROC(
            self.max_future_gain, smooth_cnt, smooth_cnt2)

        self.max_past_gain_ma = np.nan_to_num(self.max_past_gain_ma, nan=0)
        self.max_future_gain_ma = np.nan_to_num(self.max_future_gain_ma, nan=0)
        self.max_past_gain_dir = np.nan_to_num(self.max_past_gain_dir, nan=0)
        self.max_future_gain_dir = np.nan_to_num(
            self.max_future_gain_dir, nan=0)

        # Shift dir and 2nd array if necessary
        if 0 < y_lookahead_cnt:
            self.direction = np.zeros(_direction.shape)
            self.direction[:-y_lookahead_cnt] = _direction[y_lookahead_cnt:]
            self.direction = np.nan_to_num(self.direction, nan=0)

            self.directiondev2nd = np.zeros(_directiondev2nd.shape)
            self.directiondev2nd[:-
                                 y_lookahead_cnt] = _directiondev2nd[y_lookahead_cnt:]
            self.directiondev2nd = np.nan_to_num(self.directiondev2nd, nan=0)
        else:
            self.direction = _direction
            self.directiondev2nd = _directiondev2nd

        # Calculate entry and exit signals - y-type 3
        # Todo: tanh and *1000 also at __create_block__ --> Merge it to one place
        _direction_adapted = np.tanh(_direction * 1000.0)
        _directiondev2nd_adapted = np.tanh(_directiondev2nd)

        _direction_futureshifted = np.empty(_direction_adapted.shape)
        _direction_futureshifted[:] = 0.0
        _direction_futureshifted[:-
                                 self.expected_gain_lookforward] = _direction_adapted[self.expected_gain_lookforward:]

        self.entry = (_direction_adapted >= self.entr_thr) & (
            _direction_futureshifted >= self.entr_thr2) & (_directiondev2nd_adapted >= self.entr_thr3)
        self.exit = (_direction_adapted <= self.exit_thr) & (
            _direction_futureshifted <= self.exit_thr2)

        del self.gainDF

    def __next__(self):
        return self.__create_block__()

    def getCustomSizedSlice(self, custom_slice_size):
        assert 0 < custom_slice_size
        return self.__create_block__(custom_slice_size)

    def smoothMAandROC(self, arrayIn, smoothCnt, smoothCnt2=0):
        _meaned = talib.MA(arrayIn.astype(float), timeperiod=smoothCnt)

        _shifted = np.empty(_meaned.shape)
        _shifted[:] = np.nan
        _shifted[:-int(smoothCnt/2)] = _meaned[int(smoothCnt/2):]

        # Return nan if all is nan
        if all(np.isnan(_shifted)):
            return _shifted, _shifted, _shifted, _shifted

        _rocp = talib.ROCP(_shifted, timeperiod=1)

        if 1 < smoothCnt2:
            _rocp2, _rrocp, _, _ = self.smoothMAandROC(_rocp, smoothCnt2)
        else:
            _rocp2, _rrocp = np.zeros(_rocp.shape), np.zeros(_rocp.shape)

        return _shifted, _rocp, _rocp2, _rrocp

    def __create_block__(self, custom_slice_size=None):
        gain_float = []
        if custom_slice_size is None:
            _local_slice_size = self.slice_size
        else:
            _local_slice_size = custom_slice_size

        # Raise StopIteration if table is consumed
        if self.slice_start_cnt >= self.direction.shape[0]:
            logging.info(
                "Stop Iteration in Line 143 - Table consumed in y gen")
            raise StopIteration

        _dir_slice = self.direction[self.slice_start_cnt: min(
            [self.direction.shape[0], self.slice_start_cnt+_local_slice_size])]
        _dir2nddev_slice = self.directiondev2nd[self.slice_start_cnt: min(
            [self.directiondev2nd.shape[0], self.slice_start_cnt+_local_slice_size])]

        # Max gains
        _max_past_gain_slice = self.max_past_gain[self.slice_start_cnt: min(
            [self.max_past_gain.shape[0], self.slice_start_cnt+_local_slice_size])]
        _max_future_gain_slice = self.max_future_gain[self.slice_start_cnt: min(
            [self.max_future_gain.shape[0], self.slice_start_cnt+_local_slice_size])]

        _max_past_gain_ma_slice = self.max_past_gain_ma[self.slice_start_cnt: min(
            [self.max_past_gain_ma.shape[0], self.slice_start_cnt+_local_slice_size])]
        _max_future_gain_ma_slice = self.max_future_gain_ma[self.slice_start_cnt: min(
            [self.max_future_gain_ma.shape[0], self.slice_start_cnt+_local_slice_size])]

        _max_past_gain_dir_slice = self.max_past_gain_dir[self.slice_start_cnt: min(
            [self.max_past_gain_dir.shape[0], self.slice_start_cnt+_local_slice_size])]
        _max_future_gain_dir_slice = self.max_future_gain_dir[self.slice_start_cnt: min(
            [self.max_future_gain_dir.shape[0], self.slice_start_cnt+_local_slice_size])]

        gains = np.empty((_max_past_gain_slice.shape[0], 6))
        gains[:, 0] = np.tanh(_max_past_gain_slice)
        gains[:, 1] = np.tanh(_max_past_gain_ma_slice)
        gains[:, 2] = np.tanh(_max_past_gain_dir_slice)

        gains[:, 3] = np.tanh(_max_future_gain_slice)
        gains[:, 4] = np.tanh(_max_future_gain_ma_slice)
        gains[:, 5] = np.tanh(_max_future_gain_dir_slice)

        # Find rise and fall
        _falling = _dir_slice < self.fall_threshold
        _rising = (_dir_slice > self.rise_threshold) & (
            _dir2nddev_slice > 0) & (False == _falling)

        # Convert into numerical
        _falling = (_falling * -1)
        _rising = (_rising * 1)

        gainCat = _falling + _rising

        # Todo: tanh and *1000 also at entry exit calc --> Merge it to one place
        dir_float = np.empty((_dir_slice.shape[0], 2))
        dir_float[:, 0] = np.tanh(_dir_slice * 1000.0)
        dir_float[:, 1] = np.tanh(_dir2nddev_slice)

        # Entry and exit signals - y-type 3
        _entry_slice = self.entry[self.slice_start_cnt: min(
            [self.entry.shape[0], self.slice_start_cnt+_local_slice_size])]
        _exit_slice = self.exit[self.slice_start_cnt: min(
            [self.exit.shape[0], self.slice_start_cnt+_local_slice_size])]

        _signals = np.empty((_entry_slice.shape[0], 3))
        _signals[:, 0] = _entry_slice
        _signals[:, 1] = _exit_slice
        # Placeholder for "nothing" to use binary crossentropy loss
        _signals[:, 2] = np.logical_not(
            np.logical_or(_entry_slice, _exit_slice))

        self.slice_start_cnt += _local_slice_size

        # According to y_type this means: type 0, 1, 2, 3 --> Todo important: Document this!
        return gainCat, dir_float, gains, _signals


class FileListToDataStream:
    def __init__(self, fileList, batch_size, base_path,
                 smooth_cnt, smooth_cnt2, X_lookback_cnt, y_lookahead_cnt, gain_lookaround_cnt, y_type,
                 parallel_generators=4, shuffle=True, random_seed=42,
                 rise_gain_threshold=0.0, fall_gain_threshold=0.0,
                 shortspan=6, midspan=48, longspan=120,
                 y_exponent=1.0,
                 expected_gain_lookforward=48, entr_thr=0.0, entr_thr2=0.0, entr_thr3=0.0,
                 exit_thr=0.0, exit_thr2=0.0,
                 norm_price_related_indicators=True,
                 verbose=False,
                 batch_norm=True,
                 batch_norm_volume=True):

        # Todo: Provide next suitable batch sizes
        assert batch_size % parallel_generators == 0
        self.batch_size = batch_size
        self.gen_batch_size = int(batch_size / parallel_generators)
        self.parallel_generators = parallel_generators
        self.base_path = base_path
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.smooth_cnt = smooth_cnt
        self.smooth_cnt2 = smooth_cnt2
        self.y_lookahead_cnt = y_lookahead_cnt
        self.X_lookback_cnt = X_lookback_cnt
        self.gain_lookaround_cnt = gain_lookaround_cnt
        self.rise_gain_threshold = rise_gain_threshold
        self.fall_gain_threshold = fall_gain_threshold
        self.shortspan = shortspan
        self.midspan = midspan
        self.longspan = longspan
        self.y_exponent = y_exponent
        self.expected_gain_lookforward = expected_gain_lookforward
        self.entr_thr = entr_thr
        self.entr_thr2 = entr_thr2
        self.entr_thr3 = entr_thr3
        self.exit_thr = exit_thr
        self.exit_thr2 = exit_thr2
        self.norm_price_related_indicators = norm_price_related_indicators
        self.batch_norm = batch_norm
        self.batch_norm_volume = batch_norm_volume

        self.verbose = verbose

        # 0 for categorical, 1 for float, 2 for gain lookaround, 3 for entry / exit
        self.y_type = y_type

        self.X_generators = []
        self.y_generators = []

        # Init one-hot encoder
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit_transform(np.array([-1, 0, 1]).reshape(-1, 1))

        # Init IndicatorCalculator
        self.ic = IndicatorCalculator(
            self.shortspan, self.midspan, self.longspan, verbose=verbose)

        # Shuffle file list
        if self.shuffle:
            self.fileList = copy.deepcopy(sklearn.utils.shuffle(
                fileList, random_state=self.random_seed))
        else:
            self.fileList = copy.deepcopy(fileList)

        # Get split count to split file list
        file_split_cnt = int(np.ceil(len(self.fileList) / parallel_generators))
        logging.info("file_split_cnt: " + str(file_split_cnt))

        # Initalize generators
        for i in range(parallel_generators):
            _fn = self.fileList.pop()
            logging.info("File " + str(_fn) + " loaded")
            _xg, _yg = self.__initGenerators__(_fn, self.gen_batch_size)

            self.X_generators.append(_xg)
            self.y_generators.append(_yg)

        logging.info("Files left: " + str(len(self.fileList)))

    def __initGenerators__(self, fn, generator_batch_size):
        _fullname = os.path.join(self.base_path, fn)

        _tickDF = pd.read_csv(_fullname, encoding="utf-8",
                              header=0, index_col='startsAt')
        _tickDF.dropna(inplace=True)  # Does not work with cmc info being nan

        # If there is a quote volume, drop it
        if 'quoteVolume' in _tickDF.columns:
            _tickDF.drop('quoteVolume', axis=1, inplace=True)

        _indDF = self.ic.CreateAllIndicatorsTable(_tickDF)

        if True == self.norm_price_related_indicators:
            _normedDF = self.ic.NormPriceRelated(_indDF)
        else:
            _normedDF = _indDF

        _xg = XBlockGenerator(
            _normedDF, generator_batch_size, self.X_lookback_cnt, self.batch_norm, self.batch_norm_volume)
        _yg = YGainGenerator(_normedDF, generator_batch_size, self.rise_gain_threshold, self.fall_gain_threshold,
                             self.smooth_cnt, self.smooth_cnt2, self.X_lookback_cnt, self.y_lookahead_cnt, self.gain_lookaround_cnt,
                             self.expected_gain_lookforward, self.entr_thr, self.entr_thr2, self.entr_thr3, self.exit_thr, self.exit_thr2)

        return _xg, _yg

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
                            _fn, self.gen_batch_size)
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
                if _X.shape[0] != self.gen_batch_size:
                    # Check if the file list is empty
                    if 0 == len(self.fileList):
                        logging.info(
                            "Stop Iteration in Line 269 - 0 == len(self.fileList)")
                        raise StopIteration

                    _fn = self.fileList.pop()
                    logging.info("File " + str(_fn) + " loaded")
                    logging.info("Files left: " + str(len(self.fileList)))
                    self.X_generators[i], self.y_generators[i] = self.__initGenerators__(
                        _fn, self.gen_batch_size)

                    # Fill up the missing elements from the new file
                    try:
                        _missing_cnt = self.gen_batch_size - _X.shape[0]
                        _missing_X = self.X_generators[i].getCustomSizedSlice(
                            _missing_cnt)
                        _missing_y = self.y_generators[i].getCustomSizedSlice(_missing_cnt)[
                            self.y_type]

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
                            _fn, self.gen_batch_size)

                        _X = next(self.X_generators[i])
                        _y = next(self.y_generators[i])[self.y_type]

                if 0 == self.y_type:
                    assert 1 == len(_y.shape)
                elif 1 == self.y_type or 2 == self.y_type or 3 == self.y_type:
                    assert 2 == len(_y.shape)

                # Todo: Workaround to fix if shape is not (xx,1), finally check why this happens
                # if 2 != len(_y.shape):
                #  _y = _y.reshape((-1,1))

                assert _X.shape[0] == _y.shape[0]

                if _X_data is None:
                    _X_data = _X
                    _y_data = _y
                else:
                    _X_data = np.concatenate((_X_data, _X))
                    _y_data = np.concatenate((_y_data, _y))

                _y_data = _y_data**self.y_exponent

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

            if 0 == self.y_type:
                _integer_encoded = _y_data.reshape(len(_y_data), 1)
                _y_data = self.onehot_encoder.transform(
                    _integer_encoded).astype(int)

        # End if everything is consumed
        if self.batch_size != _X_data.shape[0]:
            logging.info(
                "Stop Iteration in Line 376 - self.batch_size != _X_data.shape[0]")
            logging.info("self.batch_size: " + str(self.batch_size))
            logging.info("_X_data.shape[0]: " + str(_X_data.shape[0]))
            logging.info("_X_data.shape: " + str(_X_data.shape))
            raise StopIteration

        return _X_data, _y_data
