import numpy as np
import pandas as pd
import copy
import os
import sklearn.utils
from IndicatorCalculator import IndicatorCalculator, IndicatorCalculationError
from sklearn.preprocessing import OneHotEncoder

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

GAIN_FORWARD_LOOK_CNT = 48
RISE_GAIN_THRESHOLD_PERCENT = 10 
FALL_GAIN_THRESHOLD_PERCENT = 7.5

SHORTSPAN = 6
MIDSPAN = 48
LONGSPAN = 120

COLUMNS_FOR_BATCH_NORM = ['open', 'v_AD', 'v_OBV', 'volume']
COLUMNS_FOR_BATCH_NORM_WILDCARD = ['v_ADOSC']

LOOKBACK_CNT = 256

#@title XBlockGenerator
class XBlockGenerator:
  def __init__(self, inDF, slice_size):
    self.inDF = copy.deepcopy(inDF)
    self.slice_size = slice_size
    self.slice_start_cnt = LOOKBACK_CNT
    self.i = 0
    self._col_batch_norm = None
    self._col_batch_norm_indices = []
    self._vol_col_index = []
    
    # Sort the table
    self.inDF.sort_index(inplace = True)
    
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

  def __create_block__(self, custom_slice_size = None):
    # data_X = []
    if custom_slice_size is None:
      _local_slice_size = self.slice_size
    else:      
      _local_slice_size = custom_slice_size
    
    # Raise StopIteration if table is consumed
    if self.slice_start_cnt >= self.inDFValues.shape[0]:
      logging.info("Stop Iteration in Line 74 - Table consumed in X Gen")
      raise StopIteration

    data_X = np.empty((_local_slice_size, LOOKBACK_CNT, self.inDFValues.shape[1]))
    data_X[:] = np.nan
    
    _dx_cnt = 0
    
    for self.i in range(self.slice_start_cnt,self.inDFValues.shape[0]):
      i = self.i
      _slic = copy.deepcopy(self.inDFValues[i-LOOKBACK_CNT:i,:])

      # Norm all columns that are desired for batch norm
      for ind in self._col_batch_norm_indices:
        _init_val = _slic[0, ind] #_slic.loc[np.min(_slic.index),c]
        
        if 0.0 != _init_val:
          _slic[:,ind] /= _init_val
          _slic[:,ind] -= 1.0
        else:
          _slic[:,ind] = 0.0
      
      # Norm the volume column to max == 1.0
      _vol_max = np.max(_slic[:, self._vol_col_index])
      if 0.0 < _vol_max:
        _slic[:, self._vol_col_index] /= _vol_max

      # data_X.append(_slic)
      data_X[_dx_cnt, :, :] = _slic
      _dx_cnt += 1

      # if len(data_X) == _local_slice_size:
      if _dx_cnt == _local_slice_size:
        break
    
    data_X = data_X[:_dx_cnt,:,:]

    data_X = np.nan_to_num(data_X, nan=0.0, posinf=0.0, neginf=0.0)
    # data_X = np.clip(data_X, -1000.0, 1000.0)

    self.slice_start_cnt = self.i + 1

    return data_X

#@title YGainGenerator
class YGainGenerator:
  def __init__(self, inDF, slice_size, rise_gain_threshold_percent, fall_gain_threshold_percent):
    self.gainDF = copy.deepcopy(pd.DataFrame(inDF.loc[:,'open']))
    self.slice_size = slice_size
    self.slice_start_cnt = LOOKBACK_CNT
    self.rise_gain_threshold_factor = rise_gain_threshold_percent / 100.0
    self.fall_gain_threshold_factor = fall_gain_threshold_percent / 100.0
    self.i = 0
    
    # Sort the table
    self.gainDF.sort_index(inplace = True)
    
    # Pick values
    self.gainDFValues = copy.deepcopy(self.gainDF.values)
    del self.gainDF 
 
  def __next__(self):
    return self.__create_block__()

  def getCustomSizedSlice(self, custom_slice_size):
    assert 0 < custom_slice_size
    return self.__create_block__(custom_slice_size)

  def __create_block__(self, custom_slice_size = None):
    gain_float = []
    if custom_slice_size is None:
      _local_slice_size = self.slice_size
    else:      
      _local_slice_size = custom_slice_size

    # Raise StopIteration if table is consumed
    if self.slice_start_cnt >= self.gainDFValues.shape[0]:
      logging.info("Stop Iteration in Line 143 - Table consumed in y gen")
      raise StopIteration

    for self.i in range(self.slice_start_cnt, self.gainDFValues.shape[0]):
      _g = 0
      i = self.i

      # Todo: Change from DF to array to improve speed

      if i + GAIN_FORWARD_LOOK_CNT < self.gainDFValues.shape[0]:
        #tsc = self.gainDF.index[i]
        #tsf = self.gainDF.index[i + GAIN_FORWARD_LOOK_CNT]
        cv = self.gainDFValues[i][0] #self.gainDF.loc[tsc,'open']
        fv_max = np.max(self.gainDFValues[i+1:i + GAIN_FORWARD_LOOK_CNT]) #self.gainDF.loc[tsf,'open']
        fv_min = np.min(self.gainDFValues[i+1:i + GAIN_FORWARD_LOOK_CNT])     
        
        if 0 != cv:
          _g_fall = (fv_min/cv) - 1.0
          _g_rise = (fv_max/cv) - 1.0
          
          if np.abs(_g_fall) < np.abs(_g_rise): #Rising
            _g = _g_rise
          else:
            _g = _g_fall
          
          #_g = (fv/cv) - 1.0
        else:
          logging.warning("0 == cv at i="+str(i))
        
      gain_float.append(_g)

      if len(gain_float) == _local_slice_size:
        break

    gain_float = np.array(gain_float)
    gainCat = np.zeros(gain_float.shape).astype(int)
    gainCat[gain_float < -self.fall_gain_threshold_factor] = -1
    gainCat[gain_float > self.rise_gain_threshold_factor] = 1

    self.slice_start_cnt = self.i + 1

    return gainCat, gain_float

#@title class FileListToDataStream
class FileListToDataStream:
  def __init__(self, fileList, batch_size, base_path, parallel_generators = 4, shuffle = True, random_seed=42):
    assert batch_size % parallel_generators == 0 # Todo: Provide next suitable batch sizes
    self.batch_size = batch_size
    self.gen_batch_size = int(batch_size / parallel_generators)
    self.parallel_generators = parallel_generators
    self.base_path = base_path
    self.shuffle = shuffle
    self.random_seed = random_seed

    #self.file_lists = []
    self.X_generators = []
    self.y_generators = []

    # Init one-hot encoder
    self.onehot_encoder = OneHotEncoder(sparse=False)
    self.onehot_encoder.fit_transform(np.array([-1,0,1]).reshape(-1,1))

    # Init IndicatorCalculator
    self.ic = IndicatorCalculator()

    # Shuffle file list
    if self.shuffle:
      self.fileList = copy.deepcopy(sklearn.utils.shuffle(fileList, random_state=self.random_seed))
    else:
      self.fileList = copy.deepcopy(fileList)

    # Get split count to split file list
    file_split_cnt = int(np.ceil(len(self.fileList) / parallel_generators))
    logging.info("file_split_cnt: " + str(file_split_cnt))

    # Get one file list for each generator
    #_fl = copy.deepcopy(self.fileList)
    #for i in range(parallel_generators):
    #  self.file_lists.append(_fl[:file_split_cnt])
    #  del _fl[:file_split_cnt]

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

    _tickDF = pd.read_csv(_fullname, encoding="utf-8", header=0, index_col='startsAt')
    _tickDF.dropna(inplace=True)

    # If there is a quote volume, drop it
    if 'quoteVolume' in _tickDF.columns:
      _tickDF.drop('quoteVolume', axis=1, inplace=True)

    _indDF = self.ic.CreateAllIndicatorsTable(_tickDF)
    _normedDF = self.ic.NormPriceRelated(_indDF)

    _xg = XBlockGenerator(_normedDF, generator_batch_size)
    _yg = YGainGenerator(_normedDF, generator_batch_size, RISE_GAIN_THRESHOLD_PERCENT, FALL_GAIN_THRESHOLD_PERCENT)

    return _xg, _yg

  def __next__(self):
    _X_data = None
    _y_data = None

    y_type = 0 # 0 for categorical, 1 for float 

    for i in range(self.parallel_generators):
      try:
        _X = next(self.X_generators[i])
        _y = next(self.y_generators[i])[y_type]
      except StopIteration: # If the generator stop internally
        # Check if the file list is empty
        if 0 == len(self.fileList):
          logging.info("Stop Iteration in Line 256 - 0 == len(self.fileList)")
          raise StopIteration
        
        for retry in range(10+1):
          _fn = self.fileList.pop()
          logging.info("File " + str(_fn) + " loaded")
          logging.info("Files left: " + str(len(self.fileList)))
          self.X_generators[i], self.y_generators[i] = self.__initGenerators__(_fn, self.gen_batch_size)
          try:
            _X = next(self.X_generators[i])
            _y = next(self.y_generators[i])[y_type]
            break
          except StopIteration:
            logging.warning("Stop Iteration in Line 267 on getting new generators, retry " + str(retry))
            if 10 == retry:
              raise Exception("10 == retry on getting new generators")
        
      # Check if the generator is fully consumed
      if _X.shape[0] != self.gen_batch_size:
        # Check if the file list is empty
        if 0 == len(self.fileList):
          logging.info("Stop Iteration in Line 269 - 0 == len(self.fileList)")
          raise StopIteration
        
        _fn = self.fileList.pop()
        logging.info("File " + str(_fn) + " loaded")
        logging.info("Files left: " + str(len(self.fileList)))
        self.X_generators[i], self.y_generators[i] = self.__initGenerators__(_fn, self.gen_batch_size)

        # Fill up the missing elements from the new file
        try:
          _missing_cnt = self.gen_batch_size - _X.shape[0]
          _missing_X = self.X_generators[i].getCustomSizedSlice(_missing_cnt)
          _missing_y = self.y_generators[i].getCustomSizedSlice(_missing_cnt)[y_type]
          
          _X = np.concatenate((_X, _missing_X))
          _y = np.concatenate((_y, _missing_y))
        except StopIteration:
          logging.warning("Caught StopIteration in filling missing values for " + str(_fn))
          # Check if the file list is empty
          if 0 == len(self.fileList):
            logging.info("Stop Iteration in Line 295 - 0 == len(self.fileList)")
            raise StopIteration
          
          _fn = self.fileList.pop()
          logging.info("File " + str(_fn) + " loaded")
          logging.info("Files left: " + str(len(self.fileList)))
          self.X_generators[i], self.y_generators[i] = self.__initGenerators__(_fn, self.gen_batch_size)
          
          _X = next(self.X_generators[i])
          _y = next(self.y_generators[i])[y_type]
      
      assert 1 == len(_y.shape)      
      # Todo: Workaround to fix if shape is not (xx,1), finally check why this happens
      #if 2 != len(_y.shape):
      #  _y = _y.reshape((-1,1))

      assert _X.shape[0] == _y.shape[0]

      if _X_data is None:
        _X_data = _X
        _y_data = _y
      else:
        _X_data = np.concatenate((_X_data, _X))
        _y_data = np.concatenate((_y_data, _y))

    # End if everything is consumed
    if self.batch_size != _X_data.shape[0]:
      logging.info("Stop Iteration in Line 297 - self.batch_size != _X_data.shape[0]")
      raise StopIteration

    if self.shuffle:
      _X_data, _y_data = sklearn.utils.shuffle(_X_data, _y_data, random_state=self.random_seed)

    _integer_encoded = _y_data.reshape(len(_y_data), 1)
    _y_data = self.onehot_encoder.transform(_integer_encoded).astype(int)

    return _X_data, _y_data
