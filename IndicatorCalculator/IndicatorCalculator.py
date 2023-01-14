# Version 2.0 - 2022-11-16
# - Important: Order of columns in created indicator table is changed, no compatible to V1.x versions!
# - Dynamic time span usage
# - Code cleanup
# - Remove unused TickDFToBlocks method --> Has been moved to DataStreamCreator

# Version 1.1 - 2022-02-07
# - Some Parameters moved to constructor
# Version 1 - 2021-11-11

import pandas as pd
import numpy as np
import talib
import copy


class IndicatorCalculator():
    def __init__(self, shortspan: int, midspan: int, longspan: int, **kwargs):
        '''
        The IndicatorCalculation class is used to add financial indicators to a pandas table of OHLC(V) data.
        It is based on the library "TA-Lib" (See Link https://mrjbq7.github.io/ta-lib/)

        The columns of the input table have to be named as following:
        - open, high, low, close, volume; The timestamp is the index of the table.

        ---
        # Requried constructor arguments
        - `shortspan`: An `int` defining the timespan for calculting short-term indicators (--> fast changing indicators), by default `6`
        - `midspan`: An `int` defining the timespan for calculting middle-term indicators (--> in-between changing indicators), by default `24`
        - `longspan`: An `int` defining the timespan for calculting long-term indicators (--> slowly changing indicators), by default `120`

        ---
        # Optional constructor arguments
        - `verbose`: A `bool` flag for activating printing of additional information, like table shapes. `False` by default.
        - `dropna`: A `bool` flag if rows containing `NaN` values shall be dropped. `False` by default, `NaN`s are replaced by `0.0`.
        '''

        # Class init
        self.SHORTSPAN = shortspan
        self.MIDSPAN = midspan
        self.LONGSPAN = longspan
        self.verbose = False
        self.dropna = False

        # Parse kwargs
        if "verbose" in kwargs.keys():
            if True == kwargs["verbose"]:
                self.verbose = True
        if "dropna" in kwargs.keys():
            if True == kwargs["dropna"]:
                self.dropna = True

    def CreateAllIndicatorsTable(self, sourceTable: pd.DataFrame, **kwargs):
        '''
        This method calculates all available stock indicators and merges them into one table.

        Return: A `pandas.DataFrame` containing the input `ohlcvTbl` data and all indicators. The table column names are sorted ascending.

        The naming of the returned indicator columns follows these rules:
        - The base is `INDICATOR_NAME{timeSpan}`, for example `AROONOSC24` for the Aroon Oscillator over 24 timestamps.
        - If the indicator is price-related, a trailing `c_` is include into the name. This allows normalizing these indicators relative to the assets price. Example: c_DEMA120 for the Double Exponential Moving Average over 120 timestamps
        - If it is a volume-related indicator, a trailing `vol_` is include into the name.
        - If it is a candle-pattern indicator, a trailing `pat_` is include into the name.

        Requried arguments:
        - `sourceTable`: A `pandas.DataFrame` containing these columns: `open, high, low, close`.
                    The `volume` column is optional, if it is not present, no volume-based indicators are calculated.
                    The ticks timestamps are the index of the table.

        Optional arguments:
        - `calcVolumeInidators`: A `bool` flag if volume indicators shall be calculated (Of course only if `ohlcvTbl` contains a `volume` column). `True` by default.
        - `calcPatternIndicators`: A `bool` flag if candle pattern indicators shall be calculated. `True` by default.
        - `dropNonOHLCVinputColumns`: A `bool` flag if any other columns than OHLVC shall be dropped from the input table. This is useful, as for example data from Bittrex has an QuoteVolume column in the raw data. `True` by default.
        '''

        # Parse kwargs
        calcVolumeInidators = True
        calcPatternIndicators = True
        dropNonOHLCVinputColumns = True

        if "calcVolumeInidators" in kwargs.keys():
            if False == kwargs["calcVolumeInidators"]:
                calcVolumeInidators = False
        if "calcPatternIndicators" in kwargs.keys():
            if False == kwargs["calcPatternIndicators"]:
                calcPatternIndicators = False
        if "dropNonOHLCVinputColumns" in kwargs.keys():
            if False == kwargs["dropNonOHLCVinputColumns"]:
                dropNonOHLCVinputColumns = False

        # Check if columns have to be dropped
        cleanSourceTable = sourceTable

        if True == dropNonOHLCVinputColumns:
            ALLOWED_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
            columnsToDrop = []

            for c in sourceTable:
                if not c in ALLOWED_COLUMNS:
                    columnsToDrop.append(c)

            # Drop them
            while 0 < len(columnsToDrop):
                cleanSourceTable = copy.deepcopy(cleanSourceTable)
                cleanSourceTable.drop(
                    columnsToDrop.pop(), axis=1, inplace=True)

        # Calculate non-period-sensitive indicators
        concTableRaw = pd.concat([
            cleanSourceTable,
            self.CalcCycTable(cleanSourceTable)
        ], axis=1)

        # Calculate period-sensitive indicators for each timespan
        for timeSpan in [self.SHORTSPAN, self.MIDSPAN, self.LONGSPAN]:
            # Join the new indicators into the table
            # This cannot be done using pd.concat, as some timespan-insensitve columns would be double
            concTableRaw = concTableRaw.join(self.CalcOverlapTable(
                cleanSourceTable, timeSpan), rsuffix="_dropMe")
            concTableRaw = concTableRaw.drop(
                concTableRaw.filter(regex='_dropMe').columns, axis=1)

            concTableRaw = concTableRaw.join(self.CalcMomentumTable(
                cleanSourceTable, timeSpan), rsuffix="_dropMe")
            concTableRaw = concTableRaw.drop(
                concTableRaw.filter(regex='_dropMe').columns, axis=1)

            concTableRaw = concTableRaw.join(self.CalcVolyTable(
                cleanSourceTable, timeSpan), rsuffix="_dropMe")
            concTableRaw = concTableRaw.drop(
                concTableRaw.filter(regex='_dropMe').columns, axis=1)

            concTableRaw = concTableRaw.join(self.CalcStatTable(
                cleanSourceTable, timeSpan), rsuffix="_dropMe")
            concTableRaw = concTableRaw.drop(
                concTableRaw.filter(regex='_dropMe').columns, axis=1)

        # Add volume indicators
        if ("volume" in cleanSourceTable.columns and True == calcVolumeInidators):
            concTableRaw = pd.concat([
                concTableRaw,
                self.CalcVolTable(cleanSourceTable, self.SHORTSPAN,
                                  self.MIDSPAN, self.LONGSPAN)
            ], axis=1)

        # Add pattern indicators
        if (True == calcPatternIndicators):
            concTableRaw = pd.concat([
                concTableRaw,
                self.CalcPatternTable(cleanSourceTable)
            ], axis=1)

        # Sort table columns ascending by name to ensure same output data format every time
        concTableRaw.sort_index(axis=1, inplace=True)

        if self.verbose:
            print("concTableRaw.shape = " + str(concTableRaw.shape))

        return concTableRaw

    def CalcOverlapTable(self, sourceTable: pd.DataFrame, timeSpan: int):
        '''
        Calculate the 'Overlap Studies Functions'

        https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html
        '''
        # New table to not copy the OHLCV values
        overlapTable = pd.DataFrame()

        high = sourceTable.loc[:, 'high']
        low = sourceTable.loc[:, 'low']
        close = sourceTable.loc[:, 'close']

        # Fixed period indicators
        # Todo: Change to dynamic periods
        overlapTable['c_HT_TRENDLINE'] = talib.HT_TRENDLINE(close)
        overlapTable['c_SAR_0005_005'] = talib.SAR(
            high, low, acceleration=0.005, maximum=0.05)
        overlapTable['c_SAR_002_02'] = talib.SAR(
            high, low, acceleration=0.02, maximum=0.2)

        # _mama0_0 = talib.MAMA(close, fastlimit=0, slowlimit=0)
        # _mavp30 = talib.MAVP(close, periods, minperiod=2, maxperiod=30, matype=0)
        # _sarex = SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)

        _bolushort, _bolmshort, _bollshort = talib.BBANDS(
            close, timeperiod=timeSpan, nbdevup=2, nbdevdn=2, matype=0)
        overlapTable['c_BolU{}'.format(timeSpan)] = _bolushort
        overlapTable['c_BolM{}'.format(timeSpan)] = _bolmshort
        overlapTable['c_BolL{}'.format(timeSpan)] = _bollshort

        overlapTable['c_DEMA{}'.format(timeSpan)] = talib.DEMA(
            close, timeperiod=timeSpan)
        overlapTable['c_EMA{}'.format(timeSpan)] = talib.EMA(
            close, timeperiod=timeSpan)
        overlapTable['c_KAMA{}'.format(timeSpan)] = talib.KAMA(
            close, timeperiod=timeSpan)
        overlapTable['c_MA{}'.format(timeSpan)] = talib.MA(
            close, timeperiod=timeSpan)
        overlapTable['c_MIDPOINT{}'.format(timeSpan)] = talib.MIDPOINT(
            close, timeperiod=timeSpan)
        overlapTable['c_SMA{}'.format(timeSpan)] = talib.SMA(
            close, timeperiod=timeSpan)
        overlapTable['c_TEMA{}'.format(timeSpan)] = talib.TEMA(
            close, timeperiod=timeSpan)
        overlapTable['c_TRIMA{}'.format(timeSpan)] = talib.TRIMA(
            close, timeperiod=timeSpan)
        overlapTable['c_WMA{}'.format(timeSpan)] = talib.WMA(
            close, timeperiod=timeSpan)
        overlapTable['c_MIDPRICE{}'.format(timeSpan)] = talib.MIDPRICE(
            high, low, timeperiod=timeSpan)

        if True == self.dropna:
            overlapTable.dropna(inplace=True)
        else:
            overlapTable.fillna(0, inplace=True)

        # Sort table columns ascending by name to ensure same output data format every time
        overlapTable.sort_index(axis=1, inplace=True)

        if self.verbose:
            print("overlapTable.shape = " + str(overlapTable.shape))

        return overlapTable

    def CalcMomentumTable(self, sourceTable: pd.DataFrame, timeSpan: int, ** kwargs):
        '''
        Calculate the 'Momentum Indicator Functions'

        https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
        '''
        # Parse kwargs
        calcVolumeInidators = True
        if "calcVolumeInidators" in kwargs.keys():
            if False == kwargs["calcVolumeInidators"]:
                calcVolumeInidators = False

        # New table to not copy the OHLCV values
        momentumTable = pd.DataFrame()

        open = sourceTable.loc[:, 'open']
        high = sourceTable.loc[:, 'high']
        low = sourceTable.loc[:, 'low']
        close = sourceTable.loc[:, 'close']

        if (True == calcVolumeInidators):
            vol = sourceTable.loc[:, 'volume']

            momentumTable['v_MFI{}'.format(timeSpan)] = talib.MFI(
                high, low, close, vol, timeperiod=timeSpan)

        # Fixed period indicators
        # Todo: Change to dynamic periods
        momentumTable['APO_12_26'] = talib.APO(
            close, fastperiod=12, slowperiod=26, matype=0)
        momentumTable['PPO12_26'] = talib.PPO(
            close, fastperiod=12, slowperiod=26, matype=0)
        momentumTable['BOP'] = talib.BOP(open, high, low, close)
        momentumTable['ULTOSC_7_14_28'] = talib.ULTOSC(
            high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        macd_12_26_9, macdsignal_12_26_9, macdhist_12_26_9 = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9)
        momentumTable['MACD_12_26_9'] = macd_12_26_9
        momentumTable['MACDSIGNAL_12_26_9'] = macdsignal_12_26_9
        momentumTable['MACDHIST_12_26_9'] = macdhist_12_26_9

        slowk, slowd = talib.STOCH(high, low, close, fastk_period=3,
                                   slowk_period=14, slowk_matype=0, slowd_period=3, slowd_matype=0)
        momentumTable['SLOWK'] = slowk
        momentumTable['SLOWD'] = slowd

        fastk, fastd = talib.STOCHF(
            high, low, close, fastk_period=3, fastd_period=3, fastd_matype=0)
        momentumTable['FASTK'] = fastk
        momentumTable['FASTD'] = fastd

        fastkrsi, fastdrsi = talib.STOCHRSI(
            close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        momentumTable['FASTKRSI'] = fastkrsi
        momentumTable['FASTDRSI'] = fastdrsi

        # macd, macdsignal, macdhist = MACDEXT(close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        # macd, macdsignal, macdhist = MACDFIX(close, signalperiod=9)

        # Dynamic period indicators
        momentumTable['ADX{}'.format(timeSpan)] = talib.ADX(
            high, low, close, timeperiod=timeSpan)
        momentumTable['ADXR{}'.format(timeSpan)] = talib.ADXR(
            high, low, close, timeperiod=timeSpan)
        momentumTable['AROONOSC{}'.format(timeSpan)] = talib.AROONOSC(
            high, low, timeperiod=timeSpan)
        momentumTable['CCI{}'.format(timeSpan)] = talib.CCI(
            high, low, close, timeperiod=timeSpan)
        momentumTable['CMO{}'.format(timeSpan)] = talib.CMO(
            close, timeperiod=timeSpan)
        momentumTable['DX{}'.format(timeSpan)] = talib.DX(
            high, low, close, timeperiod=timeSpan)
        momentumTable['MINUS_DI{}'.format(timeSpan)] = talib.MINUS_DI(
            high, low, close, timeperiod=timeSpan)
        momentumTable['PLUS_DI{}'.format(timeSpan)] = talib.PLUS_DI(
            high, low, close, timeperiod=timeSpan)
        momentumTable['MINUS_DM{}'.format(timeSpan)] = talib.MINUS_DM(
            high, low, timeperiod=timeSpan)
        momentumTable['PLUS_DM{}'.format(timeSpan)] = talib.PLUS_DM(
            high, low, timeperiod=timeSpan)
        momentumTable['MOM{}'.format(timeSpan)] = talib.MOM(
            close, timeperiod=timeSpan)
        momentumTable['ROC{}'.format(timeSpan)] = talib.ROC(
            close, timeperiod=timeSpan)
        momentumTable['ROCP{}'.format(timeSpan)] = talib.ROCP(
            close, timeperiod=timeSpan)
        momentumTable['ROCR{}'.format(timeSpan)] = talib.ROCR(
            close, timeperiod=timeSpan)
        momentumTable['RSI{}'.format(timeSpan)] = talib.RSI(
            close, timeperiod=timeSpan)
        momentumTable['TRIX{}'.format(timeSpan)] = talib.TRIX(
            close, timeperiod=timeSpan)
        momentumTable['WILLR{}'.format(timeSpan)] = talib.WILLR(
            high, low, close, timeperiod=timeSpan)
        aroondownshort, aroonupshort = talib.AROON(
            high, low, timeperiod=timeSpan)
        momentumTable['AROONUP{}'.format(timeSpan)] = aroonupshort
        momentumTable['AROONDOWN{}'.format(timeSpan)] = aroondownshort

        if True == self.dropna:
            momentumTable.dropna(inplace=True)
        else:
            momentumTable.fillna(0, inplace=True)

        # Sort table columns ascending by name to ensure same output data format every time
        momentumTable.sort_index(axis=1, inplace=True)

        if self.verbose:
            print("momentumTable.shape = " + str(momentumTable.shape))

        return momentumTable

    def CalcVolyTable(self, sourceTable: pd.DataFrame, timeSpan: int):
        '''
        Calculate the 'Volatility Indicator Functions'

        https://mrjbq7.github.io/ta-lib/func_groups/volatility_indicators.html
        '''
        # New table to not copy the OHLCV values
        volyTable = pd.DataFrame()

        high = sourceTable.loc[:, 'high']
        low = sourceTable.loc[:, 'low']
        close = sourceTable.loc[:, 'close']

        volyTable['TRANGE'] = talib.TRANGE(high, low, close)

        volyTable['ATR{}'.format(timeSpan)] = talib.ATR(
            high, low, close, timeperiod=timeSpan)
        volyTable['NATR{}'.format(timeSpan)] = talib.NATR(
            high, low, close, timeperiod=timeSpan)

        if True == self.dropna:
            volyTable.dropna(inplace=True)
        else:
            volyTable.fillna(0, inplace=True)

        # Sort table columns ascending by name to ensure same output data format every time
        volyTable.sort_index(axis=1, inplace=True)

        if self.verbose:
            print("volyTable.shape = " + str(volyTable.shape))

        return volyTable

    def CalcCycTable(self, sourceTable: pd.DataFrame):
        '''
        Calculate the 'Cycle Indicator Functions'

        https://mrjbq7.github.io/ta-lib/func_groups/cycle_indicators.html
        '''
        # New table to not copy the OHLCV values
        cycTable = pd.DataFrame()

        close = sourceTable.loc[:, 'close']

        # New table to not copy the tick values
        cycTable = pd.DataFrame()

        cycTable['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        cycTable['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        cycTable['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        inphase, quadrature = talib.HT_PHASOR(close)

        cycTable['INPHASE'] = inphase
        cycTable['QUADRATURE'] = quadrature

        sine, leadsine = talib.HT_SINE(close)

        cycTable['SINE'] = sine
        cycTable['LEADSINE'] = leadsine

        if True == self.dropna:
            cycTable.dropna(inplace=True)
        else:
            cycTable.fillna(0, inplace=True)

        # Sort table columns ascending by name to ensure same output data format every time
        cycTable.sort_index(axis=1, inplace=True)

        if self.verbose:
            print("cycTable.shape = " + str(cycTable.shape))

        return cycTable

    def CalcStatTable(self, sourceTable: pd.DataFrame, timeSpan: int):
        '''
        Calculate the 'Statistic Functions'

        https://mrjbq7.github.io/ta-lib/func_groups/statistic_functions.html
        '''
        # New table to not copy the OHLCV values
        statTable = pd.DataFrame()

        high = sourceTable.loc[:, 'high']
        low = sourceTable.loc[:, 'low']
        close = sourceTable.loc[:, 'close']

        # New table to not copy the tick values
        statTable = pd.DataFrame()

        statTable['BETA{}'.format(timeSpan)] = talib.BETA(
            high, low, timeperiod=timeSpan)
        statTable['CORREL{}'.format(timeSpan)] = talib.CORREL(
            high, low, timeperiod=timeSpan)
        statTable['c_LINEARREG{}'.format(timeSpan)] = talib.LINEARREG(
            close, timeperiod=timeSpan)
        statTable['LINEARREG_ANGLE{}'.format(timeSpan)] = talib.LINEARREG_ANGLE(
            close, timeperiod=timeSpan)
        statTable['c_LINEARREG_INTERCEPT{}'.format(
            timeSpan)] = talib.LINEARREG_INTERCEPT(close, timeperiod=timeSpan)
        statTable['LINEARREG_SLOPE{}'.format(timeSpan)] = talib.LINEARREG_SLOPE(
            close, timeperiod=timeSpan)
        statTable['STDDEV{}'.format(timeSpan)] = talib.STDDEV(
            close, timeperiod=timeSpan, nbdev=1)
        statTable['VAR{}'.format(timeSpan)] = talib.VAR(
            close, timeperiod=timeSpan, nbdev=1)
        statTable['c_TSF{}'.format(timeSpan)] = talib.TSF(
            close, timeperiod=timeSpan)

        if True == self.dropna:
            statTable.dropna(inplace=True)
        else:
            statTable.fillna(0, inplace=True)

        if self.verbose:
            print("statTable.shape = " + str(statTable.shape))

        # Sort table columns ascending by name to ensure same output data format every time
        statTable.sort_index(axis=1, inplace=True)

        return statTable

    def CalcVolTable(self, sourceTable: pd.DataFrame, timeSpanShort: int, timeSpanMiddle: int, timeSpanLong: int):
        '''
        Calculate the 'Volume Indicator Functions'

        https://mrjbq7.github.io/ta-lib/func_groups/volume_indicators.html

        Volume indicators have a leading `vol_` in their column name
        '''

        # New table to not copy the OHLCV values
        volTable = pd.DataFrame()

        high = sourceTable.loc[:, 'high']
        low = sourceTable.loc[:, 'low']
        close = sourceTable.loc[:, 'close']
        volume = sourceTable.loc[:, 'volume']

        volTable['v_AD'] = talib.AD(high, low, close, volume)
        volTable['v_OBV'] = talib.OBV(close, volume)

        volTable['v_ADOSC_{}_{}'.format(timeSpanShort, timeSpanMiddle)] = talib.ADOSC(
            high, low, close, volume, fastperiod=timeSpanShort, slowperiod=timeSpanMiddle)

        volTable['v_ADOSC_{}_{}'.format(timeSpanMiddle, timeSpanLong)] = talib.ADOSC(
            high, low, close, volume, fastperiod=timeSpanMiddle, slowperiod=timeSpanLong)

        volTable['v_ADOSC_{}_{}'.format(timeSpanShort, timeSpanLong)] = talib.ADOSC(
            high, low, close, volume, fastperiod=timeSpanShort, slowperiod=timeSpanLong)

        if True == self.dropna:
            volTable.dropna(inplace=True)
        else:
            volTable.fillna(0, inplace=True)

        # Sort table columns ascending by name to ensure same output data format every time
        volTable.sort_index(axis=1, inplace=True)

        if self.verbose:
            print("volTable.shape = " + str(volTable.shape))

        return volTable

    def CalcPatternTable(self, sourceTable: pd.DataFrame):
        '''
        Calculate the 'Pattern Recognition Functions'

        https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html

        Pattern indicators have a leading `pat_` in their column name
        The indicators are normed to a min/max range from `0.0` to `1.0`
        '''
        # New table to not copy the OHLCV values
        patternTable = pd.DataFrame()

        open = sourceTable.loc[:, 'open']
        high = sourceTable.loc[:, 'high']
        low = sourceTable.loc[:, 'low']
        close = sourceTable.loc[:, 'close']

        patternTable['pat_cdl2crows'] = talib.CDL2CROWS(open, high, low, close)
        patternTable['pat_cdl3blackcrows'] = talib.CDL3BLACKCROWS(
            open, high, low, close)
        patternTable['pat_cdl3inside'] = talib.CDL3INSIDE(
            open, high, low, close)
        patternTable['pat_cdl3linestrike'] = talib.CDL3LINESTRIKE(
            open, high, low, close)
        patternTable['pat_cdl3outside'] = talib.CDL3OUTSIDE(
            open, high, low, close)
        patternTable['pat_cdl3starsinsouth'] = talib.CDL3STARSINSOUTH(
            open, high, low, close)
        patternTable['pat_cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(
            open, high, low, close)
        patternTable['pat_cdlabandonedbaby'] = talib.CDLABANDONEDBABY(
            open, high, low, close, penetration=0)
        patternTable['pat_cdladvanceblock'] = talib.CDLADVANCEBLOCK(
            open, high, low, close)
        patternTable['pat_cdlbelthold'] = talib.CDLBELTHOLD(
            open, high, low, close)
        patternTable['pat_cdlbreakaway'] = talib.CDLBREAKAWAY(
            open, high, low, close)
        patternTable['pat_cdlclosingmarubozu'] = talib.CDLCLOSINGMARUBOZU(
            open, high, low, close)
        patternTable['pat_cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(
            open, high, low, close)
        patternTable['pat_cdlcounterattack'] = talib.CDLCOUNTERATTACK(
            open, high, low, close)
        patternTable['pat_cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(
            open, high, low, close, penetration=0)
        patternTable['pat_cdldoji'] = talib.CDLDOJI(open, high, low, close)
        patternTable['pat_cdldojistar'] = talib.CDLDOJISTAR(
            open, high, low, close)
        patternTable['pat_cdldragonflydoji'] = talib.CDLDRAGONFLYDOJI(
            open, high, low, close)
        patternTable['pat_cdlengulfing'] = talib.CDLENGULFING(
            open, high, low, close)
        patternTable['pat_cdleveningdojistar'] = talib.CDLEVENINGDOJISTAR(
            open, high, low, close, penetration=0)
        patternTable['pat_cdleveningstar'] = talib.CDLEVENINGSTAR(
            open, high, low, close, penetration=0)
        patternTable['pat_cdlgapsidesidewhite'] = talib.CDLGAPSIDESIDEWHITE(
            open, high, low, close)
        patternTable['pat_cdlgravestonedoji'] = talib.CDLGRAVESTONEDOJI(
            open, high, low, close)
        patternTable['pat_cdlhammer'] = talib.CDLHAMMER(open, high, low, close)
        patternTable['pat_cdlhangingman'] = talib.CDLHANGINGMAN(
            open, high, low, close)
        patternTable['pat_cdlharami'] = talib.CDLHARAMI(open, high, low, close)
        patternTable['pat_cdlharamicross'] = talib.CDLHARAMICROSS(
            open, high, low, close)
        patternTable['pat_cdlhighwave'] = talib.CDLHIGHWAVE(
            open, high, low, close)
        patternTable['pat_cdlhikkake'] = talib.CDLHIKKAKE(
            open, high, low, close)
        patternTable['pat_cdlhikkakemod'] = talib.CDLHIKKAKEMOD(
            open, high, low, close)
        patternTable['pat_cdlhomingpigeon'] = talib.CDLHOMINGPIGEON(
            open, high, low, close)
        patternTable['pat_cdlidentical3crows'] = talib.CDLIDENTICAL3CROWS(
            open, high, low, close)
        patternTable['pat_cdlinneck'] = talib.CDLINNECK(open, high, low, close)
        patternTable['pat_cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(
            open, high, low, close)
        patternTable['pat_cdlkicking'] = talib.CDLKICKING(
            open, high, low, close)
        patternTable['pat_cdlkickingbylength'] = talib.CDLKICKINGBYLENGTH(
            open, high, low, close)
        patternTable['pat_cdlladderbottom'] = talib.CDLLADDERBOTTOM(
            open, high, low, close)
        patternTable['pat_cdllongleggeddoji'] = talib.CDLLONGLEGGEDDOJI(
            open, high, low, close)
        patternTable['pat_cdllongline'] = talib.CDLLONGLINE(
            open, high, low, close)
        patternTable['pat_cdlmarubozu'] = talib.CDLMARUBOZU(
            open, high, low, close)
        patternTable['pat_cdlmatchinglow'] = talib.CDLMATCHINGLOW(
            open, high, low, close)
        patternTable['pat_cdlmathold'] = talib.CDLMATHOLD(
            open, high, low, close, penetration=0)
        patternTable['pat_cdlmorningdojistar'] = talib.CDLMORNINGDOJISTAR(
            open, high, low, close, penetration=0)
        patternTable['pat_cdlmorningstar'] = talib.CDLMORNINGSTAR(
            open, high, low, close, penetration=0)
        patternTable['pat_cdlonneck'] = talib.CDLONNECK(open, high, low, close)
        patternTable['pat_cdlpiercing'] = talib.CDLPIERCING(
            open, high, low, close)
        patternTable['pat_cdlrickshawman'] = talib.CDLRICKSHAWMAN(
            open, high, low, close)
        patternTable['pat_cdlrisefall3methods'] = talib.CDLRISEFALL3METHODS(
            open, high, low, close)
        patternTable['pat_cdlseparatinglines'] = talib.CDLSEPARATINGLINES(
            open, high, low, close)
        patternTable['pat_cdlshootingstar'] = talib.CDLSHOOTINGSTAR(
            open, high, low, close)
        patternTable['pat_cdlshortline'] = talib.CDLSHORTLINE(
            open, high, low, close)
        patternTable['pat_cdlspinningtop'] = talib.CDLSPINNINGTOP(
            open, high, low, close)
        patternTable['pat_cdlstalledpattern'] = talib.CDLSTALLEDPATTERN(
            open, high, low, close)
        patternTable['pat_cdlsticksandwich'] = talib.CDLSTICKSANDWICH(
            open, high, low, close)
        patternTable['pat_cdltakuri'] = talib.CDLTAKURI(open, high, low, close)
        patternTable['pat_cdltasukigap'] = talib.CDLTASUKIGAP(
            open, high, low, close)
        patternTable['pat_cdlthrusting'] = talib.CDLTHRUSTING(
            open, high, low, close)
        patternTable['pat_cdltristar'] = talib.CDLTRISTAR(
            open, high, low, close)
        patternTable['pat_cdlunique3river'] = talib.CDLUNIQUE3RIVER(
            open, high, low, close)
        patternTable['pat_cdlupsidegap2crows'] = talib.CDLUPSIDEGAP2CROWS(
            open, high, low, close)
        patternTable['pat_cdlxsidegap3methods'] = talib.CDLXSIDEGAP3METHODS(
            open, high, low, close)

        # As the indicators have different min/max value ranges, they are normed to a min/max range from 0.0 to 1.0
        for ci in range(patternTable.shape[1]):
            vals = patternTable.iloc[:, ci].values
            maxAbsVal = np.max(np.abs(vals))

            if 0 < maxAbsVal:
                patternTable[patternTable.columns[ci]] /= (1.0 * maxAbsVal)

            else:
                # To convert them to float
                patternTable[patternTable.columns[ci]] *= 1.0

        if True == self.dropna:
            patternTable.dropna(inplace=True)
        else:
            patternTable.fillna(0, inplace=True)

        # Sort table columns ascending by name to ensure same output data format every time
        patternTable.sort_index(axis=1, inplace=True)

        if self.verbose:
            print("patternTable.shape = " + str(patternTable.shape))

        return patternTable

    # Normalize price - related indicators
    def NormPriceRelatedIndicators(self, sourceTable, **kwargs):
        '''
        This method is used to normalize price-related indicators relative to the `open` price.
        This is useful, as the absolute prices (e.g. in USD) of several asset varies over a very large span, 
        and so do the price-related indicators, but the information inside the indicator is the same for all.

        Norming is done on the `open`, `high`, `low` and `close` column, as well as on any column which name
        starts with 'c_', indicating a price-related indicator. The `baseColumn` (by default `open`) is excepted from norming. 

        Requried arguments:
        - `sourceTable`: A `pandas.DataFrame` containing tick and indicator data.

        Optional arguments:
        - `dropBaseColumn`: A `bool` flag if the column that is normed on, the 'base' column, shall be dropped. By default, the base column is `open`. `True` by default.
        - `baseColumn`: A `string` to define the 'base' column on which the indicators shall be normed. `open` by default.
        '''

        # Parse kwargs
        dropBaseColumn = True
        baseColumn = 'open'

        if "dropBaseColumn" in kwargs.keys():
            if False == kwargs["dropBaseColumn"]:
                dropBaseColumn = False
        if "baseColumn" in kwargs.keys():
            baseColumn = str(kwargs["dropBaseColumn"])

        # Create a save copy of the table
        normedTable = copy.deepcopy(sourceTable)

        # Get the base values
        baseValues = normedTable.loc[:, baseColumn]

        # Price columns
        for c in ['open', 'close', 'high', 'low']:
            if baseColumn != c:
                normedTable[c] /= baseValues
                normedTable[c] -= 1.0

        # Iterate through other and check for beginning 'c_
        for c in normedTable.columns:
            if 'c_' == c[:2]:
                normedTable[c] /= baseValues
                normedTable[c] -= 1.0

        # Drop the base column
        if True == dropBaseColumn:
            normedTable.drop(baseColumn, axis=1, inplace=True)

        return normedTable
