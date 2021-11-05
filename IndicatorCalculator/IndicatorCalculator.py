import copy
import pandas as pd
import gc
import numpy as np
import talib
from talib import MA_Type
from sklearn.preprocessing import OneHotEncoder
    
class IndicatorCalculationError(Exception):
  def __init__(self, message):            
    # Call the base class constructor with the parameters it needs
    super().__init__(message)

class IndicatorCalculator():
  def __init__(self): # Todo: Add kwargs for parameters
    
    self.GAIN_FORWARD_LOOK_CNT = 24
    self.GAIN_THRESHOLD = 0.66
    self.GAIN_THRESHOLD_PERCENT = 10 # eliminates GAIN_THRESHOLD

    self.SHORTSPAN = 6
    self.MIDSPAN = 48
    self.LONGSPAN = 120

    self.COLUMNS_FOR_BATCH_NORM = ['open', 'v_AD', 'v_OBV', 'volume']
    self.COLUMNS_FOR_BATCH_NORM_WILDCARD = ['v_ADOSC']

    self.LOOKBACK_CNT = 128

    # Class init
    self.verbose = False
    self.calculateGain = True

    self.onehot_encoder = OneHotEncoder(sparse=False)
    self.onehot_encoder.fit_transform(np.array([-1,0,1]).reshape(-1,1))

    if self.verbose:
      from tqdm import tqdm
    else:
      def tqdm(inp): # Todo: Check for interference with external code
        return inp

    def TickDFToBlocks(self, tickDFInput):
      """
      Adds indicators to a tick table and converts it into X-blocks of len
      lookback-time and future gain as y
      """
      tickDF = copy.deepcopy(tickDFInput)
        
      # If there is a quote volume, drop it
      if 'quoteVolume' in tickDF.columns:
        tickDF.drop('quoteVolume', axis=1, inplace=True)
      
      # If there is no volume column (e.g. ticks from Coinmarketcap, add one with zeros)
      if not "volume" in tickDF.columns:
        tickDF["volume"] = 0.0

      # Calculate indicator df
      indDF = CreateAllIndicatorsTable(tickDF)

      # Raise an exception if the indicator DF is empty
      if 0 == indDF.shape[0]:
        raise IndicatorCalculationError("0 == indDF.shape[0]")

      # Norm the price-related columns
      normedDF = NormPriceRelated(indDF)
      
      # Calculate Gain
      # Important: indDF, as it is cropped to remove nans
      if calculateGain:
        gainDF = CalcGain(indDF) #Todo add self.calculateGain

        # One-Hot encode
        _gcat = gainDF.loc[:,'gainCat'].values[LOOKBACK_CNT:]
        if 0 == _gcat.shape[0]:
          raise IndicatorCalculationError("No gain data available")
        
        _integer_encoded = _gcat.reshape(len(_gcat), 1)
        y = onehot_encoder.transform(_integer_encoded).astype(int)
      else:
        y = None

      # Clean up
      del tickDF
      del indDF
      del gainDF
      gc.collect()

      # Create X-Blocks
      X = CreateXBlocks(normedDF)

      # Clean up
      del normedDF
      gc.collect()

      # Check consitency
      if X.shape[0] != y.shape[0]:
        raise IndicatorCalculationError("Different shapes of X and y!!")

      return X, y

  def CreateAllIndicatorsTable(self, tbl):
    # Concat all indicators
    concTableRaw = pd.concat([
                              tbl,
                              self.CalcOverlapTable(tbl),
                              self.CalcMomentumTable(tbl),
                              self.CalcVolyTable(tbl),
                              self.CalcCycTable(tbl),
                              self.CalcStatTable(tbl),
                              self.CalcVolTable(tbl),
                              self.CalcPatternTable(tbl)
                              ], axis=1)
    
    if self.verbose:
      print("concTableRaw.shape = " + str(concTableRaw.shape))

    return concTableRaw

  # Normalize price - related columns
  # Do this for train and test - not time-sensitive
  def NormPriceRelated(self, tbl):
    pNormed = copy.deepcopy(tbl)

    open_values = pNormed.loc[:,'open']

    #Price columns
    pNormed['close'] /= open_values
    pNormed['high'] /= open_values
    pNormed['low'] /= open_values

    pNormed['close'] -= 1.0
    pNormed['high'] -= 1.0
    pNormed['low'] -= 1.0

    #Iterate through other and check for beginning 'c_
    for c in pNormed.columns:
      if 'c_' == c[:2]: # or 'close' == c or 'high' == c or 'low' == c:
        pNormed[c] /= open_values
        pNormed[c] -= 1.0

    return pNormed

  def CalcOverlapTable(self, sourceTable):
    overlapTable = copy.deepcopy(sourceTable)
    # overlapTable.drop('Gain', axis=1, inplace=True)

    open = overlapTable.loc[:,'open']
    high = overlapTable.loc[:,'high']
    low = overlapTable.loc[:,'low']
    close = overlapTable.loc[:,'close']

    # New table to not copy the tick values
    overlapTable = pd.DataFrame()

    _bolushort, _bolmshort, _bollshort = talib.BBANDS(close, timeperiod=self.SHORTSPAN, nbdevup=2, nbdevdn=2, matype=0)
    overlapTable['c_BolU{}'.format(self.SHORTSPAN)] = _bolushort
    overlapTable['c_BolM{}'.format(self.SHORTSPAN)] = _bolmshort
    overlapTable['c_BolL{}'.format(self.SHORTSPAN)] = _bollshort

    overlapTable['c_DEMA{}'.format(self.SHORTSPAN)] =         talib.DEMA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_EMA{}'.format(self.SHORTSPAN)]  =         talib.EMA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_KAMA{}'.format(self.SHORTSPAN)] =         talib.KAMA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_MA{}'.format(self.SHORTSPAN)] =           talib.MA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_MIDPOINT{}'.format(self.SHORTSPAN)] =     talib.MIDPOINT(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_SMA{}'.format(self.SHORTSPAN)] =          talib.SMA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_TEMA{}'.format(self.SHORTSPAN)] =         talib.TEMA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_TRIMA{}'.format(self.SHORTSPAN)] =        talib.TRIMA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_WMA{}'.format(self.SHORTSPAN)] =          talib.WMA(close,timeperiod=self.SHORTSPAN)
    overlapTable['c_MIDPRICE{}'.format(self.SHORTSPAN)] =     talib.MIDPRICE(high, low, timeperiod=self.SHORTSPAN)


    _bolumid, _bolmmid, _bollmid = talib.BBANDS(close, timeperiod=self.MIDSPAN, nbdevup=2, nbdevdn=2, matype=0)
    overlapTable['c_BolU{}'.format(self.MIDSPAN)] = _bolumid
    overlapTable['c_BolM{}'.format(self.MIDSPAN)] = _bolmmid
    overlapTable['c_BolL{}'.format(self.MIDSPAN)] = _bollmid

    overlapTable['c_DEMA{}'.format(self.MIDSPAN)] =         talib.DEMA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_EMA{}'.format(self.MIDSPAN)]  =         talib.EMA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_KAMA{}'.format(self.MIDSPAN)] =         talib.KAMA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_MA{}'.format(self.MIDSPAN)] =           talib.MA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_MIDPOINT{}'.format(self.MIDSPAN)] =     talib.MIDPOINT(close,timeperiod=self.MIDSPAN)
    overlapTable['c_SMA{}'.format(self.MIDSPAN)] =          talib.SMA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_TEMA{}'.format(self.MIDSPAN)] =         talib.TEMA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_TRIMA{}'.format(self.MIDSPAN)] =        talib.TRIMA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_WMA{}'.format(self.MIDSPAN)] =          talib.WMA(close,timeperiod=self.MIDSPAN)
    overlapTable['c_MIDPRICE{}'.format(self.MIDSPAN)] =     talib.MIDPRICE(high, low, timeperiod=self.MIDSPAN)

    _bolulong, _bolmlong, _bolllong = talib.BBANDS(close, timeperiod=self.LONGSPAN, nbdevup=2, nbdevdn=2, matype=0)
    overlapTable['c_BolU{}'.format(self.LONGSPAN)] = _bolulong
    overlapTable['c_BolM{}'.format(self.LONGSPAN)] = _bolmlong
    overlapTable['c_BolL{}'.format(self.LONGSPAN)] = _bolllong

    overlapTable['c_DEMA{}'.format(self.LONGSPAN)] =         talib.DEMA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_EMA{}'.format(self.LONGSPAN)]  =         talib.EMA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_KAMA{}'.format(self.LONGSPAN)] =         talib.KAMA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_MA{}'.format(self.LONGSPAN)] =           talib.MA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_MIDPOINT{}'.format(self.LONGSPAN)] =     talib.MIDPOINT(close,timeperiod=self.LONGSPAN)
    overlapTable['c_SMA{}'.format(self.LONGSPAN)] =          talib.SMA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_TEMA{}'.format(self.LONGSPAN)] =         talib.TEMA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_TRIMA{}'.format(self.LONGSPAN)] =        talib.TRIMA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_WMA{}'.format(self.LONGSPAN)] =          talib.WMA(close,timeperiod=self.LONGSPAN)
    overlapTable['c_MIDPRICE{}'.format(self.LONGSPAN)] =     talib.MIDPRICE(high, low, timeperiod=self.LONGSPAN)


    overlapTable['c_HT_TRENDLINE'] =  talib.HT_TRENDLINE(close)
    overlapTable['c_SAR_0005_005'] = talib.SAR(high, low, acceleration=0.005, maximum=0.05)
    overlapTable['c_SAR_002_02'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    # _mama0_0 = talib.MAMA(close, fastlimit=0, slowlimit=0)
    # _mavp30 = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)
    # _sarex = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

    # overlapTable.dropna(inplace=True)
    overlapTable.fillna(0, inplace=True)

    if self.verbose:
      print("overlapTable.shape = " + str(overlapTable.shape))

    return overlapTable
  
  def CalcMomentumTable(self, sourceTable):
    momentumTable = copy.deepcopy(sourceTable)
    # momentumTable.drop('Gain', axis=1, inplace=True)

    open = momentumTable.loc[:,'open']
    high = momentumTable.loc[:,'high']
    low = momentumTable.loc[:,'low']
    close = momentumTable.loc[:,'close']
    vol = momentumTable.loc[:,'volume']

    # New table to not copy the tick values
    momentumTable = pd.DataFrame()

    momentumTable['APO_12_26'] =   talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    momentumTable['PPO12_26'] =         talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    momentumTable['BOP'] =         talib.BOP(open, high, low, close)
    momentumTable['ULTOSC_7_14_28'] =         talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)



    momentumTable['ADX{}'.format(self.SHORTSPAN)] =       talib.ADX(high, low, close, timeperiod=self.SHORTSPAN)
    momentumTable['ADXR{}'.format(self.SHORTSPAN)] =      talib.ADXR(high, low, close, timeperiod=self.SHORTSPAN)
    momentumTable['AROONOSC{}'.format(self.SHORTSPAN)] =  talib.AROONOSC(high, low, timeperiod=self.SHORTSPAN)
    momentumTable['CCI{}'.format(self.SHORTSPAN)] =       talib.CCI(high, low, close, timeperiod=self.SHORTSPAN)
    momentumTable['CMO{}'.format(self.SHORTSPAN)] =       talib.CMO(close, timeperiod=self.SHORTSPAN)
    momentumTable['DX{}'.format(self.SHORTSPAN)] =        talib.DX(high, low, close, timeperiod=self.SHORTSPAN)
    momentumTable['v_MFI{}'.format(self.SHORTSPAN)] =     talib.MFI(high, low, close, vol, timeperiod=self.SHORTSPAN)
    momentumTable['MINUS_DI{}'.format(self.SHORTSPAN)] =  talib.MINUS_DI(high, low, close, timeperiod=self.SHORTSPAN)
    momentumTable['PLUS_DI{}'.format(self.SHORTSPAN)] =   talib.PLUS_DI(high, low, close, timeperiod=self.SHORTSPAN)
    momentumTable['MINUS_DM{}'.format(self.SHORTSPAN)] =  talib.MINUS_DM(high, low, timeperiod=self.SHORTSPAN)
    momentumTable['PLUS_DM{}'.format(self.SHORTSPAN)] =   talib.PLUS_DM(high, low, timeperiod=self.SHORTSPAN)
    momentumTable['MOM{}'.format(self.SHORTSPAN)] =       talib.MOM(close, timeperiod=self.SHORTSPAN)
    momentumTable['ROC{}'.format(self.SHORTSPAN)] =       talib.ROC(close, timeperiod=self.SHORTSPAN)
    momentumTable['ROCP{}'.format(self.SHORTSPAN)] =      talib.ROCP(close, timeperiod=self.SHORTSPAN)
    momentumTable['ROCR{}'.format(self.SHORTSPAN)] =      talib.ROCR(close, timeperiod=self.SHORTSPAN)
    momentumTable['RSI{}'.format(self.SHORTSPAN)] =       talib.RSI(close, timeperiod=self.SHORTSPAN)
    momentumTable['TRIX{}'.format(self.SHORTSPAN)] =      talib.TRIX(close, timeperiod=self.SHORTSPAN)
    momentumTable['WILLR{}'.format(self.SHORTSPAN)] =     talib.WILLR(high, low, close, timeperiod=self.SHORTSPAN)

    aroondownshort, aroonupshort = talib.AROON(high, low, timeperiod=self.SHORTSPAN)
    momentumTable['AROONUP{}'.format(self.SHORTSPAN)] = aroonupshort
    momentumTable['AROONDOWN{}'.format(self.SHORTSPAN)] = aroondownshort


    momentumTable['ADX{}'.format(self.MIDSPAN)] =       talib.ADX(high, low, close, timeperiod=self.MIDSPAN)
    momentumTable['ADXR{}'.format(self.MIDSPAN)] =      talib.ADXR(high, low, close, timeperiod=self.MIDSPAN)
    momentumTable['AROONOSC{}'.format(self.MIDSPAN)] =  talib.AROONOSC(high, low, timeperiod=self.MIDSPAN)
    momentumTable['CCI{}'.format(self.MIDSPAN)] =       talib.CCI(high, low, close, timeperiod=self.MIDSPAN)
    momentumTable['CMO{}'.format(self.MIDSPAN)] =       talib.CMO(close, timeperiod=self.MIDSPAN)
    momentumTable['DX{}'.format(self.MIDSPAN)] =        talib.DX(high, low, close, timeperiod=self.MIDSPAN)
    momentumTable['v_MFI{}'.format(self.MIDSPAN)] =     talib.MFI(high, low, close, vol, timeperiod=self.MIDSPAN)
    momentumTable['MINUS_DI{}'.format(self.MIDSPAN)] =  talib.MINUS_DI(high, low, close, timeperiod=self.MIDSPAN)
    momentumTable['PLUS_DI{}'.format(self.MIDSPAN)] =   talib.PLUS_DI(high, low, close, timeperiod=self.MIDSPAN)
    momentumTable['MINUS_DM{}'.format(self.MIDSPAN)] =  talib.MINUS_DM(high, low, timeperiod=self.MIDSPAN)
    momentumTable['PLUS_DM{}'.format(self.MIDSPAN)] =   talib.PLUS_DM(high, low, timeperiod=self.MIDSPAN)
    momentumTable['MOM{}'.format(self.MIDSPAN)] =       talib.MOM(close, timeperiod=self.MIDSPAN)
    momentumTable['ROC{}'.format(self.MIDSPAN)] =       talib.ROC(close, timeperiod=self.MIDSPAN)
    momentumTable['ROCP{}'.format(self.MIDSPAN)] =      talib.ROCP(close, timeperiod=self.MIDSPAN)
    momentumTable['ROCR{}'.format(self.MIDSPAN)] =      talib.ROCR(close, timeperiod=self.MIDSPAN)
    momentumTable['RSI{}'.format(self.MIDSPAN)] =       talib.RSI(close, timeperiod=self.MIDSPAN)
    momentumTable['TRIX{}'.format(self.MIDSPAN)] =      talib.TRIX(close, timeperiod=self.MIDSPAN)
    momentumTable['WILLR{}'.format(self.MIDSPAN)] =     talib.WILLR(high, low, close, timeperiod=self.MIDSPAN)

    aroondownmid, aroonupmid = talib.AROON(high, low, timeperiod=self.MIDSPAN)
    momentumTable['AROONUP{}'.format(self.MIDSPAN)] = aroonupmid
    momentumTable['AROONDOWN{}'.format(self.MIDSPAN)] = aroondownmid


    momentumTable['ADX{}'.format(self.LONGSPAN)] =       talib.ADX(high, low, close, timeperiod=self.LONGSPAN)
    momentumTable['ADXR{}'.format(self.LONGSPAN)] =      talib.ADXR(high, low, close, timeperiod=self.LONGSPAN)
    momentumTable['AROONOSC{}'.format(self.LONGSPAN)] =  talib.AROONOSC(high, low, timeperiod=self.LONGSPAN)
    momentumTable['CCI{}'.format(self.LONGSPAN)] =       talib.CCI(high, low, close, timeperiod=self.LONGSPAN)
    momentumTable['CMO{}'.format(self.LONGSPAN)] =       talib.CMO(close, timeperiod=self.LONGSPAN)
    momentumTable['DX{}'.format(self.LONGSPAN)] =        talib.DX(high, low, close, timeperiod=self.LONGSPAN)
    momentumTable['v_MFI{}'.format(self.LONGSPAN)] =     talib.MFI(high, low, close, vol, timeperiod=self.LONGSPAN)
    momentumTable['MINUS_DI{}'.format(self.LONGSPAN)] =  talib.MINUS_DI(high, low, close, timeperiod=self.LONGSPAN)
    momentumTable['PLUS_DI{}'.format(self.LONGSPAN)] =   talib.PLUS_DI(high, low, close, timeperiod=self.LONGSPAN)
    momentumTable['MINUS_DM{}'.format(self.LONGSPAN)] =  talib.MINUS_DM(high, low, timeperiod=self.LONGSPAN)
    momentumTable['PLUS_DM{}'.format(self.LONGSPAN)] =   talib.PLUS_DM(high, low, timeperiod=self.LONGSPAN)
    momentumTable['MOM{}'.format(self.LONGSPAN)] =       talib.MOM(close, timeperiod=self.LONGSPAN)
    momentumTable['ROC{}'.format(self.LONGSPAN)] =       talib.ROC(close, timeperiod=self.LONGSPAN)
    momentumTable['ROCP{}'.format(self.LONGSPAN)] =      talib.ROCP(close, timeperiod=self.LONGSPAN)
    momentumTable['ROCR{}'.format(self.LONGSPAN)] =      talib.ROCR(close, timeperiod=self.LONGSPAN)
    momentumTable['RSI{}'.format(self.LONGSPAN)] =       talib.RSI(close, timeperiod=self.LONGSPAN)
    momentumTable['TRIX{}'.format(self.LONGSPAN)] =      talib.TRIX(close, timeperiod=self.LONGSPAN)
    momentumTable['WILLR{}'.format(self.LONGSPAN)] =     talib.WILLR(high, low, close, timeperiod=self.LONGSPAN)

    aroondownlong, aroonuplong = talib.AROON(high, low, timeperiod=self.LONGSPAN)
    momentumTable['AROONUP{}'.format(self.LONGSPAN)] = aroonuplong
    momentumTable['AROONDOWN{}'.format(self.LONGSPAN)] = aroondownlong


    macd_12_26_9, macdsignal_12_26_9, macdhist_12_26_9 = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    momentumTable['MACD_12_26_9'] = macd_12_26_9
    momentumTable['MACDSIGNAL_12_26_9'] = macdsignal_12_26_9
    momentumTable['MACDHIST_12_26_9'] = macdhist_12_26_9

    #macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    #macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)

    slowk, slowd = talib.STOCH(high, low, close, fastk_period=3, slowk_period=14, slowk_matype=0, slowd_period=3, slowd_matype=0)
    momentumTable['SLOWK'] = slowk
    momentumTable['SLOWD'] = slowd

    fastk, fastd = talib.STOCHF(high, low, close, fastk_period=3, fastd_period=3, fastd_matype=0)
    momentumTable['FASTK'] = fastk
    momentumTable['FASTD'] = fastd

    fastkrsi, fastdrsi = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    momentumTable['FASTKRSI'] = fastkrsi
    momentumTable['FASTDRSI'] = fastdrsi

    # momentumTable.dropna(inplace=True)
    momentumTable.fillna(0, inplace=True)

    if self.verbose:
      print("momentumTable.shape = " + str(momentumTable.shape))

    return momentumTable

  def CalcVolyTable(self, sourceTable): 
    volyTable = copy.deepcopy(sourceTable)
    # volyTable.drop('Gain', axis=1, inplace=True)

    open = volyTable.loc[:,'open']
    high = volyTable.loc[:,'high']
    low = volyTable.loc[:,'low']
    close = volyTable.loc[:,'close'] 

    # New table to not copy the tick values
    volyTable = pd.DataFrame()

    volyTable['TRANGE'] = talib.TRANGE(high, low, close)

    volyTable['ATR{}'.format(self.SHORTSPAN)] =  talib.ATR(high, low, close, timeperiod=self.SHORTSPAN)
    volyTable['NATR{}'.format(self.SHORTSPAN)] = talib.NATR(high, low, close, timeperiod=self.SHORTSPAN)

    volyTable['ATR{}'.format(self.MIDSPAN)] =  talib.ATR(high, low, close, timeperiod=self.MIDSPAN)
    volyTable['NATR{}'.format(self.MIDSPAN)] = talib.NATR(high, low, close, timeperiod=self.MIDSPAN)

    volyTable['ATR{}'.format(self.LONGSPAN)] =  talib.ATR(high, low, close, timeperiod=self.LONGSPAN)
    volyTable['NATR{}'.format(self.LONGSPAN)] = talib.NATR(high, low, close, timeperiod=self.LONGSPAN)

    # volyTable.dropna(inplace=True)
    volyTable.fillna(0, inplace=True)
    
    if self.verbose:
      print("volyTable.shape = " + str(volyTable.shape))

    return volyTable

  def CalcCycTable(self, sourceTable): 
    cycTable = copy.deepcopy(sourceTable)
    # cycTable.drop('Gain', axis=1, inplace=True)

    open = cycTable.loc[:,'open']
    high = cycTable.loc[:,'high']
    low = cycTable.loc[:,'low']
    close = cycTable.loc[:,'close']

    # New table to not copy the tick values
    cycTable = pd.DataFrame()

    cycTable['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    cycTable['HT_DCPHASE'] =  talib.HT_DCPHASE(close)
    cycTable['HT_TRENDMODE'] =  talib.HT_TRENDMODE(close)

    inphase, quadrature = talib.HT_PHASOR(close)

    cycTable['INPHASE'] = inphase
    cycTable['QUADRATURE'] = quadrature

    sine, leadsine = talib.HT_SINE(close)

    cycTable['SINE'] = sine
    cycTable['LEADSINE'] = leadsine

    # cycTable.dropna(inplace=True)
    cycTable.fillna(0, inplace=True)
    
    if self.verbose:
      print("cycTable.shape = " + str(cycTable.shape))

    return cycTable

  def CalcStatTable(self, sourceTable): 
    statTable = copy.deepcopy(sourceTable)
    # statTable.drop('Gain', axis=1, inplace=True)

    open = statTable.loc[:,'open']
    high = statTable.loc[:,'high']
    low = statTable.loc[:,'low']
    close = statTable.loc[:,'close']

    # New table to not copy the tick values
    statTable = pd.DataFrame()

    statTable['BETA{}'.format(self.SHORTSPAN)] =                   talib.BETA(high, low, timeperiod=self.SHORTSPAN)
    statTable['CORREL{}'.format(self.SHORTSPAN)] =                 talib.CORREL(high, low, timeperiod=self.SHORTSPAN)
    statTable['c_LINEARREG{}'.format(self.SHORTSPAN)] =            talib.LINEARREG(close, timeperiod=self.SHORTSPAN)
    statTable['LINEARREG_ANGLE{}'.format(self.SHORTSPAN)] =        talib.LINEARREG_ANGLE(close, timeperiod=self.SHORTSPAN)
    statTable['c_LINEARREG_INTERCEPT{}'.format(self.SHORTSPAN)] =  talib.LINEARREG_INTERCEPT(close, timeperiod=self.SHORTSPAN)
    statTable['LINEARREG_SLOPE{}'.format(self.SHORTSPAN)] =        talib.LINEARREG_SLOPE(close, timeperiod=self.SHORTSPAN)
    statTable['STDDEV{}'.format(self.SHORTSPAN)] =                 talib.STDDEV(close, timeperiod=self.SHORTSPAN, nbdev=1)
    statTable['VAR{}'.format(self.SHORTSPAN)] =                    talib.VAR(close, timeperiod=self.SHORTSPAN, nbdev=1)
    statTable['c_TSF{}'.format(self.SHORTSPAN)] =                  talib.TSF(close, timeperiod=self.SHORTSPAN)

    statTable['BETA{}'.format(self.MIDSPAN)] =                   talib.BETA(high, low, timeperiod=self.MIDSPAN)
    statTable['CORREL{}'.format(self.MIDSPAN)] =                 talib.CORREL(high, low, timeperiod=self.MIDSPAN)
    statTable['c_LINEARREG{}'.format(self.MIDSPAN)] =            talib.LINEARREG(close, timeperiod=self.MIDSPAN)
    statTable['LINEARREG_ANGLE{}'.format(self.MIDSPAN)] =        talib.LINEARREG_ANGLE(close, timeperiod=self.MIDSPAN)
    statTable['c_LINEARREG_INTERCEPT{}'.format(self.MIDSPAN)] =  talib.LINEARREG_INTERCEPT(close, timeperiod=self.MIDSPAN)
    statTable['LINEARREG_SLOPE{}'.format(self.MIDSPAN)] =        talib.LINEARREG_SLOPE(close, timeperiod=self.MIDSPAN)
    statTable['STDDEV{}'.format(self.MIDSPAN)] =                 talib.STDDEV(close, timeperiod=self.MIDSPAN, nbdev=1)
    statTable['VAR{}'.format(self.MIDSPAN)] =                    talib.VAR(close, timeperiod=self.MIDSPAN, nbdev=1)
    statTable['c_TSF{}'.format(self.MIDSPAN)] =                  talib.TSF(close, timeperiod=self.MIDSPAN)

    statTable['BETA{}'.format(self.LONGSPAN)] =                   talib.BETA(high, low, timeperiod=self.LONGSPAN)
    statTable['CORREL{}'.format(self.LONGSPAN)] =                 talib.CORREL(high, low, timeperiod=self.LONGSPAN)
    statTable['c_LINEARREG{}'.format(self.LONGSPAN)] =            talib.LINEARREG(close, timeperiod=self.LONGSPAN)
    statTable['LINEARREG_ANGLE{}'.format(self.LONGSPAN)] =        talib.LINEARREG_ANGLE(close, timeperiod=self.LONGSPAN)
    statTable['c_LINEARREG_INTERCEPT{}'.format(self.LONGSPAN)] =  talib.LINEARREG_INTERCEPT(close, timeperiod=self.LONGSPAN)
    statTable['LINEARREG_SLOPE{}'.format(self.LONGSPAN)] =        talib.LINEARREG_SLOPE(close, timeperiod=self.LONGSPAN)
    statTable['STDDEV{}'.format(self.LONGSPAN)] =                 talib.STDDEV(close, timeperiod=self.LONGSPAN, nbdev=1)
    statTable['VAR{}'.format(self.LONGSPAN)] =                    talib.VAR(close, timeperiod=self.LONGSPAN, nbdev=1)
    statTable['c_TSF{}'.format(self.LONGSPAN)] =                  talib.TSF(close, timeperiod=self.LONGSPAN)

    # statTable.dropna(inplace=True)
    statTable.fillna(0, inplace=True)
    
    if self.verbose:
      print("statTable.shape = " + str(statTable.shape))

    return statTable

  def CalcVolTable(self, sourceTable): 
    volTable = copy.deepcopy(sourceTable)
    # volTable.drop('Gain', axis=1, inplace=True)

    high = volTable.loc[:,'high']
    low = volTable.loc[:,'low']
    close = volTable.loc[:,'close']
    volume = volTable.loc[:,'volume']

    # New table to not copy the tick values
    volTable = pd.DataFrame()

    volTable['v_AD'] =         talib.AD(high, low, close, volume)
    volTable['v_ADOSC_{}_{}'.format(self.SHORTSPAN, self.MIDSPAN)] = talib.ADOSC(high, low, close, volume, fastperiod=self.SHORTSPAN, slowperiod=self.MIDSPAN)
    volTable['v_ADOSC_{}_{}'.format(self.MIDSPAN, self.LONGSPAN)] = talib.ADOSC(high, low, close, volume, fastperiod=self.MIDSPAN, slowperiod=self.LONGSPAN)
    volTable['v_ADOSC_{}_{}'.format(self.SHORTSPAN, self.LONGSPAN)] = talib.ADOSC(high, low, close, volume, fastperiod=self.SHORTSPAN, slowperiod=self.LONGSPAN)
    volTable['v_ADOSC_3_10'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    volTable['v_OBV'] =        talib.OBV(close, volume)

    # volTable.dropna(inplace=True)
    volTable.fillna(0, inplace=True)

    if self.verbose:
      print("volTable.shape = " + str(volTable.shape))

    return volTable

  def CalcPatternTable(self, sourceTable):
    patternTable = copy.deepcopy(sourceTable)
    
    open = patternTable.loc[:,'open']
    high = patternTable.loc[:,'high']
    low = patternTable.loc[:,'low']
    close = patternTable.loc[:,'close']

    # New table to not copy the tick values
    patternTable = pd.DataFrame()

    patternTable['pat_cdl2crows'] = talib.CDL2CROWS(open, high, low, close)
    patternTable['pat_cdl3blackcrows'] = talib.CDL3BLACKCROWS(open, high, low, close)
    patternTable['pat_cdl3inside'] = talib.CDL3INSIDE(open, high, low, close)
    patternTable['pat_cdl3linestrike'] = talib.CDL3LINESTRIKE(open, high, low, close)
    patternTable['pat_cdl3outside'] = talib.CDL3OUTSIDE(open, high, low, close)
    patternTable['pat_cdl3starsinsouth'] = talib.CDL3STARSINSOUTH(open, high, low, close)
    patternTable['pat_cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(open, high, low, close)
    patternTable['pat_cdlabandonedbaby'] = talib.CDLABANDONEDBABY(open, high, low, close, penetration =0)
    patternTable['pat_cdladvanceblock'] = talib.CDLADVANCEBLOCK(open, high, low, close)
    patternTable['pat_cdlbelthold'] = talib.CDLBELTHOLD(open, high, low, close)
    patternTable['pat_cdlbreakaway'] = talib.CDLBREAKAWAY(open, high, low, close)
    patternTable['pat_cdlclosingmarubozu'] = talib.CDLCLOSINGMARUBOZU(open, high, low, close)
    patternTable['pat_cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(open, high, low, close)
    patternTable['pat_cdlcounterattack'] = talib.CDLCOUNTERATTACK(open, high, low, close)
    patternTable['pat_cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
    patternTable['pat_cdldoji'] = talib.CDLDOJI(open, high, low, close)
    patternTable['pat_cdldojistar'] = talib.CDLDOJISTAR(open, high, low, close)
    patternTable['pat_cdldragonflydoji'] = talib.CDLDRAGONFLYDOJI(open, high, low, close)
    patternTable['pat_cdlengulfing'] = talib.CDLENGULFING(open, high, low, close)
    patternTable['pat_cdleveningdojistar'] = talib.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)
    patternTable['pat_cdleveningstar'] = talib.CDLEVENINGSTAR(open, high, low, close, penetration=0)
    patternTable['pat_cdlgapsidesidewhite'] = talib.CDLGAPSIDESIDEWHITE(open, high, low, close)
    patternTable['pat_cdlgravestonedoji'] = talib.CDLGRAVESTONEDOJI(open, high, low, close)
    patternTable['pat_cdlhammer'] = talib.CDLHAMMER(open, high, low, close)
    patternTable['pat_cdlhangingman'] = talib.CDLHANGINGMAN(open, high, low, close)
    patternTable['pat_cdlharami'] = talib.CDLHARAMI(open, high, low, close)
    patternTable['pat_cdlharamicross'] = talib.CDLHARAMICROSS(open, high, low, close)
    patternTable['pat_cdlhighwave'] = talib.CDLHIGHWAVE(open, high, low, close)
    patternTable['pat_cdlhikkake'] = talib.CDLHIKKAKE(open, high, low, close)
    patternTable['pat_cdlhikkakemod'] = talib.CDLHIKKAKEMOD(open, high, low, close)
    patternTable['pat_cdlhomingpigeon'] = talib.CDLHOMINGPIGEON(open, high, low, close)
    patternTable['pat_cdlidentical3crows'] = talib.CDLIDENTICAL3CROWS(open, high, low, close)
    patternTable['pat_cdlinneck'] = talib.CDLINNECK(open, high, low, close)
    patternTable['pat_cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(open, high, low, close)
    patternTable['pat_cdlkicking'] = talib.CDLKICKING(open, high, low, close)
    patternTable['pat_cdlkickingbylength'] = talib.CDLKICKINGBYLENGTH(open, high, low, close)
    patternTable['pat_cdlladderbottom'] = talib.CDLLADDERBOTTOM(open, high, low, close)
    patternTable['pat_cdllongleggeddoji'] = talib.CDLLONGLEGGEDDOJI(open, high, low, close)
    patternTable['pat_cdllongline'] = talib.CDLLONGLINE(open, high, low, close)
    patternTable['pat_cdlmarubozu'] = talib.CDLMARUBOZU(open, high, low, close)
    patternTable['pat_cdlmatchinglow'] = talib.CDLMATCHINGLOW(open, high, low, close)
    patternTable['pat_cdlmathold'] = talib.CDLMATHOLD(open, high, low, close, penetration=0)
    patternTable['pat_cdlmorningdojistar'] = talib.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
    patternTable['pat_cdlmorningstar'] = talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)
    patternTable['pat_cdlonneck'] = talib.CDLONNECK(open, high, low, close)
    patternTable['pat_cdlpiercing'] = talib.CDLPIERCING(open, high, low, close)
    patternTable['pat_cdlrickshawman'] = talib.CDLRICKSHAWMAN(open, high, low, close)
    patternTable['pat_cdlrisefall3methods'] = talib.CDLRISEFALL3METHODS(open, high, low, close)
    patternTable['pat_cdlseparatinglines'] = talib.CDLSEPARATINGLINES(open, high, low, close)
    patternTable['pat_cdlshootingstar'] = talib.CDLSHOOTINGSTAR(open, high, low, close)
    patternTable['pat_cdlshortline'] = talib.CDLSHORTLINE(open, high, low, close)
    patternTable['pat_cdlspinningtop'] = talib.CDLSPINNINGTOP(open, high, low, close)
    patternTable['pat_cdlstalledpattern'] = talib.CDLSTALLEDPATTERN(open, high, low, close)
    patternTable['pat_cdlsticksandwich'] = talib.CDLSTICKSANDWICH(open, high, low, close)
    patternTable['pat_cdltakuri'] = talib.CDLTAKURI(open, high, low, close)
    patternTable['pat_cdltasukigap'] = talib.CDLTASUKIGAP(open, high, low, close)
    patternTable['pat_cdlthrusting'] = talib.CDLTHRUSTING(open, high, low, close)
    patternTable['pat_cdltristar'] = talib.CDLTRISTAR(open, high, low, close)
    patternTable['pat_cdlunique3river'] = talib.CDLUNIQUE3RIVER(open, high, low, close)
    patternTable['pat_cdlupsidegap2crows'] = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
    patternTable['pat_cdlxsidegap3methods'] = talib.CDLXSIDEGAP3METHODS(open, high, low, close)

    # Norm range to a min/max of 1.0
    for ci in range(patternTable.shape[1]):
      vals = patternTable.iloc[:,ci].values
      maxAbsVal = np.max(np.abs(vals))

      if 0 < maxAbsVal:
        patternTable.iloc[:,ci] /= (1.0 * maxAbsVal)
      else:
        patternTable.iloc[:,ci] *= 1.0 # To convert them to float

    patternTable.fillna(0, inplace=True)
    
    if self.verbose:
      print("patternTable.shape = " + str(patternTable.shape))

    return patternTable