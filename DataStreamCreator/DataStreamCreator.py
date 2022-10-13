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
import sklearn.utils
from IndicatorCalculator import IndicatorCalculator, IndicatorCalculationError
from sklearn.preprocessing import OneHotEncoder
import talib

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

COLUMNS_FOR_BATCH_NORM = ['v_AD', 'v_OBV', 'volume'] # Todo: Removed 'open', different behaviour in older applications now
COLUMNS_FOR_BATCH_NORM_WILDCARD = ['v_ADOSC']


class XBlockGenerator:
    def __init__(self, inDF, slice_size, X_lookback_cnt, batch_norm, batch_norm_volume):
        self.inDF = copy.deepcopy(inDF)
        self.slice_size = slice_size
        self.X_lookback_cnt = X_lookback_cnt
        # Attention: The value of self.slice_start_cnt changes during iteration, self.X_lookback_cnt does not
        self.slice_start_cnt = X_lookback_cnt
        self.i = 0
        self._col_batch_norm = None
        self._col_batch_norm_indices = []
        self._vol_col_index = []
        self.batch_norm = batch_norm
        self.batch_norm_volume = batch_norm_volume

        # Sort the table
        self.inDF.sort_index(inplace=True)

        # Find batch norm columns names
        self._col_batch_norm = copy.deepcopy(COLUMNS_FOR_BATCH_NORM)
        for c in self.inDF.columns:
            for wc in COLUMNS_FOR_BATCH_NORM_WILDCARD:
                if wc in c:
                    self._col_batch_norm.append(c)

        # Find batch norm columns indices
        for c in self._col_batch_norm:
            self._col_batch_norm_indices.append(
                list(self.inDF.columns).index(c)
            )

        # Find volume column index
        if 'volume' in self.inDF.columns:
            self._vol_col_index = list(self.inDF.columns).index('volume')

        self.inDFValues = copy.deepcopy(self.inDF.values)
        del self.inDF

    def __next__(self):
        return self.__create_block__()

    def getCustomSizedSlice(self, custom_slice_size):
        assert 0 < custom_slice_size
        return self.__create_block__(custom_slice_size)

    def __create_block__(self, custom_slice_size=None):
        # data_X = []
        if custom_slice_size is None:
            _local_slice_size = self.slice_size
        else:
            _local_slice_size = custom_slice_size

        # Raise StopIteration if table is consumed
        if self.slice_start_cnt >= self.inDFValues.shape[0]:
            logging.info("Stop Iteration in Line 74 - Table consumed in X Gen")
            raise StopIteration

        data_X = np.empty(
            (_local_slice_size, self.X_lookback_cnt, self.inDFValues.shape[1]))
        data_X[:] = np.nan

        _dx_cnt = 0

        for self.i in range(self.slice_start_cnt, self.inDFValues.shape[0]):
            i = self.i
            _slic = copy.deepcopy(self.inDFValues[i-self.X_lookback_cnt:i, :])

            # Norm all columns that are desired for batch norm
            if True == self.batch_norm:
                for ind in self._col_batch_norm_indices:
                    _init_val = _slic[0, ind]  # _slic.loc[np.min(_slic.index),c]

                    if 0.0 != _init_val:
                        _slic[:, ind] /= _init_val
                        _slic[:, ind] -= 1.0
                    else:
                        _slic[:, ind] = 0.0

            # Norm the volume column to max == 1.0
            if True == self.batch_norm_volume:
                _vol_max = np.max(_slic[:, self._vol_col_index])
                if 0.0 < _vol_max:
                    _slic[:, self._vol_col_index] /= _vol_max

            # data_X.append(_slic)
            data_X[_dx_cnt, :, :] = _slic
            _dx_cnt += 1

            # if len(data_X) == _local_slice_size:
            if _dx_cnt == _local_slice_size:
                break

        data_X = data_X[:_dx_cnt, :, :]

        data_X = np.nan_to_num(data_X, nan=0.0, posinf=0.0, neginf=0.0)
        # data_X = np.clip(data_X, -1000.0, 1000.0)

        self.slice_start_cnt = self.i + 1

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
            self.shortspan, self.midspan, self.longspan, self.verbose)

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
