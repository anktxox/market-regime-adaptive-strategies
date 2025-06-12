import pandas as pd
import numpy as np
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
from pprint import pprint
import plotly.graph_objects as go
import numpy as np
import datetime
import os
import sys
os.chdir(r'C:\Users\harsh\Desktop\inter-iit\untrade-sdk')
untrade1= os.path.abspath('untrade-sdk')
sys.path.append(untrade1)
from untrade.client import Client
client = Client()
import pandas_ta as ta
from filterpy.kalman import KalmanFilter




def process_data(filepath_4h,filepath_1D):

    def process_eth1_data(data):
        def eth_kalman(data,start_date='2020-01-01', sl_percent=5,tp_percent=25, volume_rolling_window=14, kf_R=100, kf_Q_diag=(0.001, 0.001),stc_threshold=50):
            # Load data
            df = data.copy()
            df.index = pd.to_datetime(df['datetime'])
            df.drop('datetime', axis=1, inplace=True)
            prices = df['close'].values

            # Kalman Filter smoothing
            def kalman_filter(data):
                kf = KalmanFilter(dim_x=2, dim_z=1)  # 2 states (price & velocity), 1 measurement
                kf.x = np.array([data[0], 0])  # Initial state (price, velocity)
                kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
                kf.H = np.array([[1, 0]])  # Measurement matrix
                kf.P *= 1000  # Initial state covariance matrix
                kf.R = kf_R  # Measurement noise
                kf.Q = np.array([[kf_Q_diag[0], 0], [0, kf_Q_diag[1]]])  # Process noise covariance
                smoothed_prices = []
                for price in data:
                    kf.predict()
                    kf.update(price)
                    smoothed_prices.append(kf.x[0])
                return smoothed_prices

            smoothed_prices = kalman_filter(prices)

            def calculate_stc(df, short_period=30, long_period=50, stc_period=15):
                # Calculate short-term and long-term EMAs
                df['short_ema'] = df['close'].ewm(span=short_period, adjust=False).mean().shift(1)
                df['long_ema'] = df['close'].ewm(span=long_period, adjust=False).mean().shift(1)
                # Calculate MACD
                df['macd'] = df['short_ema'] - df['long_ema']
                # Calculate %K using shifted rolling min/max
                df['%K'] = ((df['macd'] - df['macd'].rolling(stc_period).min().shift(1)) /
                            (df['macd'].rolling(stc_period).max().shift(1) - df['macd'].rolling(stc_period).min().shift(1))) * 100
                # Calculate %D as the rolling mean of %K
                df['%D'] = df['%K'].rolling(window=6).mean().shift(1)
                # Calculate STC using the EMA of %D
                df['STC'] = df['%D'].ewm(span=5, adjust=False).mean().shift(1)
                return df

            df = calculate_stc(df)

            # Generate signals
            def generate_signals_and_trade_type(df,sl_percent,tp_percent,start_date, stc_threshold=50):
                current_position = None
                entry_price = None
                tp = None
                sl = None
                signals = [0] * len(df)
                start_idx = df.index.get_loc(start_date)
                rolling_volume = df['volume'].rolling(window=volume_rolling_window).mean()
                # Iterate over the DataFrame from the start index
                for i in range(start_idx, len(df)):
                    vol_condition = df['volume'].iloc[i] >= rolling_volume.iloc[i]
                    stc_bullish = df['STC'].iloc[i - 1] > stc_threshold
                    stc_bearish = df['STC'].iloc[i - 1] < (100 - stc_threshold)

                    if current_position == 'long':
                        if df['high'].iloc[i] >= tp or df['low'].iloc[i] <= sl:
                            signals[i] = -1  # Exit long
                            current_position = None
                            entry_price = None
                            tp = None
                            sl = None

                    elif current_position == 'short':
                        if df['high'].iloc[i] >= sl or df['low'].iloc[i] <= tp:
                            signals[i] = 1  # Exit short
                            current_position = None
                            entry_price = None
                            tp = None
                            sl = None

                    # Bullish
                    if df['close'].iloc[i] > df['smoothed_price'].iloc[i]:
                        if stc_bullish and vol_condition:
                            if current_position == 'short':
                                signals[i] = 2  # Short exit & long entry
                            elif current_position is None:
                                signals[i] = 1  # Long entry
                            current_position = 'long'
                            entry_price = df['close'].iloc[i]
                            tp = entry_price * (1 + tp_percent / 100)
                            sl = entry_price * (1 - sl_percent / 100)

                    # Bearish
                    elif df['close'].iloc[i] < df['smoothed_price'].iloc[i]:
                        if stc_bearish and vol_condition:
                            if current_position == 'long':
                                signals[i] = -2  # Long exit & short entry
                            elif current_position is None:
                                signals[i] = -1  # Short entry
                            current_position = 'short'
                            entry_price = df['close'].iloc[i]
                            tp = entry_price * (1 - tp_percent / 100)
                            sl = entry_price * (1 + sl_percent / 100)

                    # Ensure no consecutive identical signals
                    if i > start_idx and signals[i] == signals[i - 1]:
                        signals[i] = 0
                # Add signals to DataFrame
                df['signals'] = signals

                return df
            

            def assign_trades(df, signal_column='signals'):
                trade_s = False  # Short position status
                trade_l = False  # Long position status
                df['trade_type'] = ''
                for index, row in df.iterrows():
                    signal = row[signal_column]
                    if signal == 1:
                        if not trade_s and not trade_l:
                            df.at[index, 'trade_type'] = 'long'
                            trade_l = True
                        elif trade_s and not trade_l:
                            df.at[index, 'trade_type'] = 'no_trade'
                            trade_s = False
                        elif not trade_s and trade_l:
                            df.at[index, 'trade_type'] = 'long'
                    elif signal == -1:
                        if not trade_s and not trade_l:
                            df.at[index, 'trade_type'] = 'short'
                            trade_s = True
                        elif not trade_s and trade_l:
                            df.at[index, 'trade_type'] = 'no_trade'
                            trade_l = False
                        elif trade_s and not trade_l:
                            df.at[index, 'trade_type'] = 'short'
                    elif signal == 2:
                        df.at[index, 'trade_type'] = 'long'
                        trade_s = False
                        trade_l = True
                    elif signal == -2:
                        df.at[index, 'trade_type'] = 'short'
                        trade_s = True
                        trade_l = False
                    elif signal == 0:
                        if trade_s and not trade_l:
                            df.at[index, 'trade_type'] = 'short'
                        elif not trade_s and trade_l:
                            df.at[index, 'trade_type'] = 'long'
                        else:
                            df.at[index, 'trade_type'] = ''
                return df

            df['smoothed_price'] = smoothed_prices
            df = generate_signals_and_trade_type(df,sl_percent,tp_percent,start_date,stc_threshold)
            df = df['2020-01-01':]
            df = assign_trades(df)
            df['datetime'] = df.index
            df = df[['datetime','close','signals','trade_type']]
            return df

        start_date = '2020-01-01'
        sl_percent = 7
        tp_percent = 100
        volume_rolling_window = 7
        kf_R = 100
        kf_Q_diag = (0.001, 0.001)
        stc = 50

        data = eth_kalman(data,start_date,sl_percent=sl_percent,tp_percent=tp_percent,volume_rolling_window=volume_rolling_window,kf_R=kf_R, kf_Q_diag=kf_Q_diag,stc_threshold=stc)
        return data


    def process_eth2_data(data):
        def process_data(df):
            def calculate_rsi(df, period=30):
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].shift(1)
                return df

            def calculate_stc(df, short_period=30, long_period=50, stc_period=14):
                df['short_ema'] = df['close'].ewm(span=short_period, adjust=False).mean()
                df['long_ema'] = df['close'].ewm(span=long_period, adjust=False).mean()
                df['macd'] = df['short_ema'] - df['long_ema']
                df['%K'] = ((df['macd'] - df['macd'].rolling(stc_period).min()) /
                            (df['macd'].rolling(stc_period).max() - df['macd'].rolling(stc_period).min())) * 100
                df['%D'] = df['%K'].rolling(window=6).mean()
                df['STC'] = df['%D'].ewm(span=3, adjust=False).mean().shift(1)
                return df

            def calculate_awesome_oscillator(df):
                df['midpoint'] = (df['high'] + df['low']) / 2
                df['sma_5'] = df['midpoint'].rolling(window=5).mean()
                df['sma_34'] = df['midpoint'].rolling(window=34).mean()
                df['awesome_oscillator'] = df['sma_5'] - df['sma_34']
                df['awesome_oscillator'] = df['awesome_oscillator'].shift(1)
                df['prev'] = df['awesome_oscillator'].shift(1)
                return df

            df['EMA_35'] = df['close'].ewm(span=35, adjust=False).mean()
            df['EMA_35'] = df['EMA_35'].shift(1)
            df = calculate_awesome_oscillator(df)

            df['ATRr_14'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            df['CHOP'] = ta.chop(high=df['high'], low=df['low'], close=df['close'], length=14, append=True)
            df['Choppy_Market'] = (df['CHOP'] > 60)

            df = calculate_rsi(df)
            df = calculate_stc(df)
            df = df.tail(1462)
            return df

        def strat(df):
            df = df.copy()
            df['signals'] = 0
            trade_long = False
            trade_short = False
            df['trade_type'] = 'Hold'

            for index, row in df.iterrows():
                price = row['close']
                ema_35 = row['EMA_35']
                stc = row['STC']
                rsi = row['RSI']
                ao = row['awesome_oscillator']
                ao1 = row['prev']
                chop = row['Choppy_Market']
                atr = row['ATRr_14']

                if ((price > ema_35 and stc >= 50 and ao > ao1 and ao > 0)) and not trade_long and not chop:
                    if trade_short:
                        df.at[index, 'signals'] = 2
                    else:
                        df.at[index, 'signals'] = 1
                    trade_long = True
                    trade_short = False

                elif ((price < ema_35 and stc <= 50 and ao < ao1 and ao < 0)) and not trade_short and not chop:
                    if trade_long:
                        df.at[index, 'signals'] = -2
                    else:
                        df.at[index, 'signals'] = -1
                    trade_long = False
                    trade_short = True

                elif trade_long and ((rsi < 40) or stc < 20):
                    df.at[index, 'signals'] = -1
                    trade_long = False

                elif trade_short and ((rsi > 50) or stc > 80):
                    df.at[index, 'signals'] = 1
                    trade_short = False

            df["TP"] = np.where(df["signals"] > 0, np.minimum(df["close"] + 2 * df['ATRr_14'], df["close"] * 1.3), 0)
            df["TP"] = np.where(df["signals"] < 0, np.maximum(df["close"] - 2 * df['ATRr_14'], df["close"] * 0.7), 0)
            df["SL"] = np.where(df["signals"] > 0, np.minimum(df["close"] - 1.5 * df['ATRr_14'], df["close"] * 0.8), 0)
            df["SL"] = np.where(df["signals"] < 0, np.maximum(df["close"] + 1.5 * df['ATRr_14'], df["close"] * 1.2), 0)
            return df

        def trade_gen(df):
            trade_s = False
            trade_l = False
            df['trade'] = 'no_trade'
            for index, row in df.iterrows():
                if row['signals'] == 1:
                    if not trade_s and not trade_l:
                        df.at[index, 'trade'] = 'long'
                        trade_l = True
                    elif trade_s and not trade_l:
                        df.at[index, 'trade'] = 'no_trade'
                        trade_s = False
                    elif not trade_s and trade_l:
                        df.at[index, 'trade'] = 'long'
                elif row['signals'] == -1:
                    if not trade_s and not trade_l:
                        df.at[index, 'trade'] = 'short'
                        trade_s = True
                    elif not trade_s and trade_l:
                        df.at[index, 'trade'] = 'no_trade'
                        trade_l = False
                    elif trade_s and not trade_l:
                        df.at[index, 'trade'] = 'short'
                elif row['signals'] == 2:
                    df.at[index, 'trade'] = 'long'
                    trade_s = False
                    trade_l = True
                elif row['signals'] == -2:
                    df.at[index, 'trade'] = 'short'
                    trade_s = True
                    trade_l = False
                elif row['signals'] == 0:
                    if trade_s and not trade_l:
                        df.at[index, 'trade'] = 'short'
                    elif not trade_s and trade_l:
                        df.at[index, 'trade'] = 'long'
                else:
                    df.at[index, 'trade'] = 'no_trade'
            return df

        # Full processing and strategy
        processed_data = process_data(data)
        strategy_data = strat(processed_data)
        final_data = trade_gen(strategy_data)
        return final_data[['datetime', 'close', 'signals', 'trade_type', 'TP', 'SL']]

    def process_eth3_data(filepath):
        # Load data
        data = pd.read_csv(filepath)
        # Calculate indicators
        data.ta.chop(high='high', close='close', low='low', append=True)
        data.ta.aroon(high='high', low='low', length=28, append=True)
        data.ta.adx(high='high', low='low', close='close', length=28, append=True)
        data.ta.adx(high='high', low='low', close='close', length=14, append=True)
        data.ta.macd(close='close', append=True)
        data.ta.rsi(close='close', append=True)
        data.ta.atr(high='high', low='low', close='close', length=14, append=True)
        data['ATR_MA'] = data['ATRr_14'].rolling(window=14).mean()

        # Heiken Ashi conversion
        ha_df = data.copy()
        ha_df['HA_close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
        ha_df['HA_open'] = (ha_df['open'] + ha_df['close']) / 2
        for i in range(1, len(ha_df)):
            ha_df.at[i, 'HA_open'] = (ha_df.at[i-1, 'HA_open'] + ha_df.at[i-1, 'HA_close']) / 2
        ha_df['HA_high'] = ha_df[['high', 'HA_open', 'HA_close']].max(axis=1)
        ha_df['HA_low'] = ha_df[['low', 'HA_open', 'HA_close']].min(axis=1)
        data['HA'] = ha_df.apply(lambda row: 1 if row['HA_open'] < row['HA_close'] else -1, axis=1)

        # Define choppy and stagnant market
        choppy_market = (
            (data['ADX_14'] < 20) &
            (data['ATRr_14'] < data['ATR_MA']) &
            (data['RSI_14'].between(40, 60))
        )
        data['Choppy'] = choppy_market
        data['Rolling_Std'] = data['close'].rolling(window=28).std()
        data['Normalized_Std'] = data['Rolling_Std'] / data['close']
        data['Stagnant_Market'] = data['Normalized_Std'] < 0.005
        data['Stagnant_Market'] = data['Stagnant_Market'].fillna(False)

        # Calculate signals
        def total_signal(df, current_candle):
            current_pos = df.index.get_loc(current_candle)
            c1 = df['high'].iloc[current_pos] > df['high'].iloc[current_pos - 1]
            c2 = df['high'].iloc[current_pos - 1] > df['low'].iloc[current_pos]
            c3 = df['low'].iloc[current_pos] > df['high'].iloc[current_pos - 2]
            c4 = df['high'].iloc[current_pos - 2] > df['low'].iloc[current_pos - 1]
            c5 = df['low'].iloc[current_pos - 1] > df['high'].iloc[current_pos - 3]
            c6 = df['high'].iloc[current_pos - 3] > df['low'].iloc[current_pos - 2]
            c7 = df['low'].iloc[current_pos - 2] > df['low'].iloc[current_pos - 3]

            if c1 and c2 and c3 and c4 and c5 and c6 and c7 and df['HA'].iloc[current_pos] == 1 and \
            df['AROONOSC_28'].iloc[current_pos] > 0 and \
            df['DMP_28'].iloc[current_pos] >= df['DMN_28'].iloc[current_pos] and \
            df['MACDh_12_26_9'].iloc[current_pos] >= df['MACDh_12_26_9'].iloc[current_pos - 1]:
                return 1

            c1 = df['low'].iloc[current_pos] < df['low'].iloc[current_pos - 1]
            c2 = df['low'].iloc[current_pos - 1] < df['high'].iloc[current_pos]
            c3 = df['high'].iloc[current_pos] < df['low'].iloc[current_pos - 2]
            c4 = df['low'].iloc[current_pos - 2] < df['high'].iloc[current_pos - 1]
            c5 = df['high'].iloc[current_pos - 1] < df['low'].iloc[current_pos - 3]
            c6 = df['low'].iloc[current_pos - 3] < df['high'].iloc[current_pos - 2]
            c7 = df['high'].iloc[current_pos - 2] < df['high'].iloc[current_pos - 3]

            if c1 and c2 and c3 and c4 and c5 and c6 and c7 and df['HA'].iloc[current_pos] == -1 and \
            df['AROONOSC_28'].iloc[current_pos] < 0 and \
            df['DMP_28'].iloc[current_pos] <= df['DMN_28'].iloc[current_pos] and \
            df['MACDh_12_26_9'].iloc[current_pos] <= df['MACDh_12_26_9'].iloc[current_pos - 1]:
                return -1
            return 0

        data['signals'] = data.apply(lambda row: total_signal(data, row.name), axis=1)
        data.loc[data['Choppy'] == True, 'signals'] = 0
        data.loc[data['Stagnant_Market'] == True, 'signals'] = 0

        # Define TP and SL
        data['TP'] = np.where(data['signals'] > 0, data['close'] + 2 * data['ATRr_14'],
                            np.where(data['signals'] < 0, data['close'] - 2 * data['ATRr_14'], 0))
        data['SL'] = np.where(data['signals'] > 0, data['close'] - 1.5 * data['ATRr_14'],
                            np.where(data['signals'] < 0, data['close'] + 1.5 * data['ATRr_14'], 0))

        # Trade management
        trade_s = False
        trade_l = False
        data=data.copy()
        data['trade'] = ''

        for index, row in data.iterrows():
            if row['signals'] == 1:
                if not trade_s and not trade_l:
                    data.at[index, 'trade'] = 'long'
                    trade_l = True
                elif trade_s and not trade_l:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_s = False
                    trade_l = True
                elif not trade_s and trade_l:
                    data.at[index, 'trade'] = 'long'
            elif row['signals'] == -1:
                if not trade_s and not trade_l:
                    data.at[index, 'trade'] = 'short'
                    trade_s = True
                elif not trade_s and trade_l:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_l = False
                    trade_s = True
                elif trade_s and not trade_l:
                    data.at[index, 'trade'] = 'short'
            elif row['signals'] == 2 and trade_l:
                data.at[index, 'trade'] = 'no_trade'
                trade_l = False
            elif row['signals'] == -2 and trade_s:
                data.at[index, 'trade'] = 'no_trade'
                trade_s = False
            elif row['signals'] == 0:
                if trade_s:
                    data.at[index, 'trade'] = 'short'
                elif trade_l:
                    data.at[index, 'trade'] = 'long'
                else:
                    data.at[index, 'trade'] = 'no_trade'
        data['trade_type'] = data['trade']
        data = data[['datetime', 'close', 'TP', 'SL', 'signals', 'trade_type']]

        # Return processed data
        return data
    
    def process_eth4_data(data_path):
        # Load data
        data = pd.read_csv(data_path)

        # Preprocessing
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data[data['datetime'] >= '2020-01-01 00:00:00']
        data = data[data['volume'] != 0]
        data = data.reset_index(drop=True)

        # Calculate indicators
        data.ta.chop(high='high', close='close', low='low', append=True)
        data.ta.aroon(high='high', low='low', length=28, append=True)
        data.ta.rsi(close='close', append=True)
        data.ta.adx(high='high', low='low', close='close', append=True)
        data.ta.atr(high='high', low='low', close='close', append=True)
        data['ATR_MA'] = data['ATRr_14'].rolling(window=14).mean()

        # Define choppy market condition
        choppy_market = (
            (data['ADX_14'] < 20) &
            (data['ATRr_14'] < data['ATR_MA']) &
            (data['RSI_14'].between(40, 60))
        )
        data['Choppy'] = choppy_market

        # Compute rolling standard deviation
        data['Rolling_Std'] = data['close'].rolling(window=20).std()
        data['Normalized_Std'] = data['Rolling_Std'] / data['close']
        threshold = 0.010  # Adjust based on strategy
        data['Stagnant_Market'] = data['Normalized_Std'] < threshold
        data['Stagnant_Market'] = data['Stagnant_Market'].fillna(False)

        # Signal generation function
        def total_signal(df, current_candle):
            current_pos = df.index.get_loc(current_candle)

            # Buy condition
            c1 = df['high'].iloc[current_pos] > df['high'].iloc[current_pos - 1]
            c2 = df['high'].iloc[current_pos - 1] > df['low'].iloc[current_pos]
            c3 = df['low'].iloc[current_pos] > df['high'].iloc[current_pos - 2]
            c4 = df['high'].iloc[current_pos - 2] > df['low'].iloc[current_pos - 1]
            c5 = df['low'].iloc[current_pos - 1] > df['high'].iloc[current_pos - 3]
            c6 = df['high'].iloc[current_pos - 3] > df['low'].iloc[current_pos - 2]
            c7 = df['low'].iloc[current_pos - 2] > df['low'].iloc[current_pos - 3]

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                return 1

            # Sell condition
            c1 = df['low'].iloc[current_pos] < df['low'].iloc[current_pos - 1]
            c2 = df['low'].iloc[current_pos - 1] < df['high'].iloc[current_pos]
            c3 = df['high'].iloc[current_pos] < df['low'].iloc[current_pos - 2]
            c4 = df['low'].iloc[current_pos - 2] < df['high'].iloc[current_pos - 1]
            c5 = df['high'].iloc[current_pos - 1] < df['low'].iloc[current_pos - 3]
            c6 = df['low'].iloc[current_pos - 3] < df['high'].iloc[current_pos - 2]
            c7 = df['high'].iloc[current_pos - 2] < df['high'].iloc[current_pos - 3]

            if c1 and c2 and c3 and c4 and c5 and c6 and c7:
                return -1

            return 0

        # Apply signals
        data['signals'] = data.apply(lambda row: total_signal(data, row.name), axis=1)
        data.loc[data['Choppy'] | data['Stagnant_Market'], 'signals'] = 0

        # Strategy logic
        data["TP"] = np.where(data["signals"] > 0, data["close"] * 1.3, 0)
        data["TP"] = np.where(data["signals"] < 0, data["close"] * 0.7, 0)
        data["SL"] = np.where(data["signals"] > 0, data["close"] * 0.9, 0)
        data["SL"] = np.where(data["signals"] < 0, data["close"] * 1.1, 0)

        trade_s = False
        trade_l = False
        data['trade'] = ''

        for index, row in data.iterrows():
            if row['signals'] == 1:  # long signal
                if not trade_s and not trade_l:
                    data.at[index, 'trade'] = 'long'
                    trade_l = True
                elif trade_s and not trade_l:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_s = False
                    trade_l = True
                elif not trade_s and trade_l:
                    data.at[index, 'trade'] = 'long'

            elif row['signals'] == -1:  # short signal
                if not trade_s and not trade_l:
                    data.at[index, 'trade'] = 'short'
                    trade_s = True
                elif not trade_s and trade_l:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_l = False
                    trade_s = True
                elif trade_s and not trade_l:
                    data.at[index, 'trade'] = 'short'

            elif row['signals'] == 2:  # close long
                if trade_l:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_l = False

            elif row['signals'] == -2:  # close short
                if trade_s:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_s = False

            elif row['signals'] == 0:
                if trade_s:
                    data.at[index, 'trade'] = 'short'
                elif trade_l:
                    data.at[index, 'trade'] = 'long'
                else:
                    data.at[index, 'trade'] = 'no_trade'
        data['trade_type'] = data['trade']
        data = data[['close', 'datetime', 'TP', 'SL', 'signals', 'trade_type']]
        return data
    btc_1d = pd.read_csv(filepath_1D)
    strat1 = process_eth1_data(btc_1d)
    strat2 = process_eth2_data(btc_1d)
    strat3 = process_eth3_data(filepath_4h)
    strat4 = process_eth4_data(filepath_4h)

    return strat1,strat2,strat3,strat4



def strat(strat1,strat2,strat3,strat4):
    strat3['datetime'] = pd.to_datetime(strat3['datetime'])
    strat3.index = strat3['datetime']
    strat2['datetime'] = pd.to_datetime(strat2['datetime'])
    new_index2 = pd.date_range(start=strat2['datetime'].min(), end=strat2['datetime'].max(), freq="4H")
    strat2 = strat2.set_index('datetime').reindex(new_index2)
    strat2['close'] = strat3['close']
    strat2['signals'] = strat2['signals'].fillna(0)
    strat2['trade_type'] = strat2['trade_type'].fillna(method = 'ffill')
    strat2['TP'] = strat2['TP'].fillna(0)
    strat2['SL'] = strat2['SL'].fillna(0)
    strat1['datetime'] = pd.to_datetime(strat1['datetime'])
    new_index1 = pd.date_range(start=strat1['datetime'].min(), end=strat1['datetime'].max(), freq="4H")
    strat1 = strat1.set_index('datetime').reindex(new_index1)
    strat1['close'] = strat3['close']
    strat1['signals'] = strat1['signals'].fillna(0)
    strat1['trade_type'] = strat1['trade_type'].fillna(method = 'ffill')
    strat3['datetime'] = pd.to_datetime(strat3['datetime'])
    strat3.index = strat3['datetime']
    strat4['datetime'] = pd.to_datetime(strat4['datetime'])
    strat4.index = strat4['datetime']
    strat4['close'] = strat1['close']

    def calculate_booksize(df):
        df['close_change'] = df['close']-df['close'].shift(1)
        df['booksize'] = np.nan
        df['booksize'][0] = 0
        for i in range(1,len(df)):
            if df['signals'][i]==1:
                while i<len(df) and df['signals'][i]!=-1:
                    df['booksize'][i] = df['close_change'][i]+df['booksize'][i-1]
                    i+=1
                if i<len(df):
                    df['booksize'][i] = df['close_change'][i] + df['booksize'][i-1]
            elif df['signals'][i]==-1:
                while i<len(df) and df['signals'][i]!=1:
                    df['booksize'][i] = -df['close_change'][i]+df['booksize'][i-1]
                    i+=1
                if i<len(df):
                    df['booksize'][i] = -df['close_change'][i]+df['booksize'][i-1]
            else:
                df['booksize'][i] = df['booksize'][i-1]

    calculate_booksize(strat1)
    calculate_booksize(strat2)
    calculate_booksize(strat3)
    calculate_booksize(strat4)
    final = pd.DataFrame()
    final.index = strat1.index
    final['close'] = strat1['close']
    final['sig1'] = strat1['signals']
    final['trade_type1'] = strat1['trade_type']
    final['TP1'] = np.zeros(len(strat1))
    final['SL1'] = np.zeros(len(strat1))
    final['booksize1'] = strat1['booksize']
    final['sig2'] = strat2['signals']
    final['trade_type2'] = strat2['trade_type']
    final['booksize2'] = strat2['booksize']
    final['TP2'] = strat2['TP']
    final['SL2'] = strat2['SL']
    final['sig3'] = strat3['signals']
    final['trade_type3'] = strat3['trade_type']
    final['booksize3'] = strat3['booksize']
    final['TP3'] = strat3['TP']
    final['SL3'] = strat3['TP']
    final['sig4'] = strat4['signals']
    final['trade_type4'] = strat4['trade_type']
    final['booksize4'] = strat4['booksize']
    final['TP4'] = np.zeros(len(strat1))
    final['SL4'] = np.zeros(len(strat1))


    def best_strategy(df, i, win,weights=None):
        if weights is None:
            weights = {
                'volatility': 0.1,
                'mean_pos_ret': 0.05,
                'max_drawdown': 0.1,
                'sharpe_ratio': 0.7,
                'alpha': 0.05
            }

        rank = OrderedDict()

        # Compute percentage change (returns) for each booksize
        returns1 = df['booksize1'][max(0, i-win+1):i].pct_change().replace([np.nan], 0).fillna(0)
        returns2 = df['booksize2'][max(0, i-win+1):i].pct_change().replace([np.nan], 0).fillna(0)
        returns3 = df['booksize3'][max(0, i-win+1):i].pct_change().replace([np.nan], 0).fillna(0)
        returns4 = df['booksize4'][max(0, i-win+1):i].pct_change().replace([np.nan], 0).fillna(0)

        # Standard deviation of returns (volatility)
        volatility1 = returns1.rolling(win-1).std().iloc[-1] if not returns1.empty else 0
        volatility1 = 0 if pd.isna(volatility1) else volatility1
        volatility2 = returns2.rolling(win-1).std().iloc[-1] if not returns2.empty else 0
        volatility2 = 0 if pd.isna(volatility2) else volatility2
        volatility3 = returns3.rolling(win-1).std().iloc[-1] if not returns3.empty else 0
        volatility3 = 0 if pd.isna(volatility3) else volatility3
        volatility4 = returns4.rolling(win-1).std().iloc[-1] if not returns4.empty else 0
        volatility4 = 0 if pd.isna(volatility4) else volatility4

        # Mean of positive returns (average return)
        mean_pos_ret1 = returns1[returns1 > 0].mean() if not returns1[returns1 > 0].empty else 0
        mean_pos_ret2 = returns2[returns2 > 0].mean() if not returns2[returns2 > 0].empty else 0
        mean_pos_ret3 = returns3[returns3 > 0].mean() if not returns3[returns3 > 0].empty else 0
        mean_pos_ret4 = returns4[returns4 > 0].mean() if not returns4[returns4 > 0].empty else 0

        # Maximum Drawdown (based on the cumulative maximum)
        max_drawdown1 = (df['booksize1'][max(0, i-win+1):i].cummax() - df['booksize1'][max(0, i-win+1):i]).max()
        max_drawdown2 = (df['booksize2'][max(0, i-win+1):i].cummax() - df['booksize2'][max(0, i-win+1):i]).max()
        max_drawdown3 = (df['booksize3'][max(0, i-win+1):i].cummax() - df['booksize3'][max(0, i-win+1):i]).max()
        max_drawdown4 = (df['booksize4'][max(0, i-win+1):i].cummax() - df['booksize4'][max(0, i-win+1):i]).max()

        # Compute risk-adjusted return (Sharpe ratio)
        sharpe1 = mean_pos_ret1 / volatility1 if volatility1 != 0 else 0
        sharpe2 = mean_pos_ret2 / volatility2 if volatility2 != 0 else 0
        sharpe3 = mean_pos_ret3 / volatility3 if volatility3 != 0 else 0
        sharpe4 = mean_pos_ret4 / volatility4 if volatility4 != 0 else 0



        # Rank each strategy based on weighted score
        rank[1] = (
            weights['volatility'] * volatility1 +
            weights['mean_pos_ret'] * mean_pos_ret1 +
            weights['max_drawdown'] * max_drawdown1 +
            weights['sharpe_ratio'] * sharpe1 
        )
        
        rank[2] = (
            weights['volatility'] * volatility2 +
            weights['mean_pos_ret'] * mean_pos_ret2 +
            weights['max_drawdown'] * max_drawdown2 +
            weights['sharpe_ratio'] * sharpe2 
        )
        
        rank[3] = (
            weights['volatility'] * volatility3 +
            weights['mean_pos_ret'] * mean_pos_ret3 +
            weights['max_drawdown'] * max_drawdown3 +
            weights['sharpe_ratio'] * sharpe3 
        )

        rank[4] = (
            weights['volatility'] * volatility4 +
            weights['mean_pos_ret'] * mean_pos_ret4 +
            weights['max_drawdown'] * max_drawdown4 +
            weights['sharpe_ratio'] * sharpe4
        )
        sorted_rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        best = sorted_rank[0][0]

        return best
    

    def sig_to_trade(df):
        trade_s = False
        trade_l = False
        df['trade_type'] = ''
        for index, row in df.iterrows():
            if(row['signals'] == 1):
                if trade_s==False and trade_l==False:
                    df.at[index, 'trade_type'] = 'long'
                    trade_l = True
                elif trade_s==True and trade_l==False:
                    df.at[index, 'trade_type'] = 'close'
                    trade_s = False


            elif(row['signals'] == -1):
                if trade_s==False and trade_l==False:
                    df.at[index, 'trade_type'] = 'short'
                    trade_s = True
                elif trade_s == False and trade_l == True:
                    df.at[index, 'trade_type'] = 'close'
                    trade_l = False

            elif(row['signals'] == 2):
                df.at[index, 'trade_type'] = 'long_reversal'
                trade_s = False
                trade_l = True
            elif(row['signals'] == -2):
                df.at[index, 'trade_type'] = 'short_reversal'
                trade_s = True
                trade_l = False
        for index, row in df.iterrows():
            if df.at[index, 'trade_type'] == '':
                df.at[index, 'signals'] = 0
                
                    
    def master_strategy(df,win,small_win):
        best_strat = 1
        df['final_sig'] = np.nan
        df['final_trade'] = np.nan
        df['best'] = np.nan
        df['TP'] = np.zeros(len(df))
        df['SL'] = np.zeros(len(df))
        df['ch'] = np.nan
        i = 0
        while i<len(df):
            if i==0:
                df['final_sig'][i:i+win] = df[f'sig{best_strat}'][i:i+win]
                df['final_trade'][i:i+win] = df[f'trade_type{best_strat}'][i:i+win]
                df['best'][i:i+win] = best_strat
                i+=win-1
                continue
            best_strat = best_strategy(df,i,win)
            df['ch'][i] = 'yes'
            if best_strat == 0 :
                if df['final_trade'][i-1] == 'long':
                    df['final_sig'][i] = -1
                if df['final_trade'][i-1] == 'short':
                    df['final_sig'][i] = 1
                if df['final_trade'][i-1] == 'no_trade':
                    df['final_sig'][i] = 0
                df['final_trade'][i] = "no_trade"
                i+=1
                while best_strategy(df,i,small_win) == 0:
                    df['final_sig'][i] = 0
                    df['final_trade'][i] = "no_trade"
                    i+=1
                best_strat = best_strategy(df,i,small_win)
                df['final_sig'][i:i+small_win] = df[f'sig{best_strat}'][i:i+small_win]
                df['final_trade'][i:i+small_win] = df[f'trade_type{best_strat}'][i:i+small_win]
                df['TP'][i:i+small_win] = df[f'TP{best_strat}'][i:i+small_win]
                df['SL'][i:i+small_win] = df[f'SL{best_strat}'][i:i+small_win]
                df['best'][i:i+small_win] = best_strat
                i+=small_win-1
                continue

            if df['final_trade'][i-1]=='long' and df[f'trade_type{best_strat}'][i]=='long':
                df['final_sig'][i] = 0
                df['final_trade'][i] = "long"
            elif df['final_trade'][i-1]=='long' and df[f'trade_type{best_strat}'][i]=='short':
                df['final_sig'][i]  = -2
                df['final_trade'][i] = "short"
            elif df['final_trade'][i-1]=='long' and df[f'trade_type{best_strat}'][i]=='no_trade':
                df['final_sig'][i]  = -1
                df['final_trade'][i] = "no_trade"
            elif df['final_trade'][i-1]=='short' and df[f'trade_type{best_strat}'][i]=='short':
                df['final_sig'][i]  = 0
                df['final_trade'][i] = "short"
            elif df['final_trade'][i-1]=='short' and df[f'trade_type{best_strat}'][i]=='no_trade':
                df['final_sig'][i]  = 1
                df['final_trade'][i] = "no_trade"
            elif df['final_trade'][i-1]=='short' and df[f'trade_type{best_strat}'][i]=='long':
                df['final_sig'][i]  = 2
                df['final_trade'][i] = "long"
            elif df['final_trade'][i-1]=='no_trade' and df[f'trade_type{best_strat}'][i]=='short':
                df['final_sig'][i]  = -1
                df['final_trade'][i] = "short"
            elif df['final_trade'][i-1]=='no_trade' and df[f'trade_type{best_strat}'][i]=='no_trade':
                df['final_sig'][i]  = 0
                df['final_trade'][i] = "no_trade"
            elif df['final_trade'][i-1]=='no_trade' and df[f'trade_type{best_strat}'][i]=='long':
                df['final_sig'][i]  = 1
                df['final_trade'][i] = "long"
            df['final_sig'][i+1:i+win+1] = df[f'sig{best_strat}'][i+1:i+win+1]
            df['final_trade'][i+1:i+win+1] = df[f'trade_type{best_strat}'][i+1:i+win+1]
            df['TP'][i+1:i+win+1] = df[f'TP{best_strat}'][i+1:i+win+1]
            df['SL'][i+1:i+win+1] = df[f'SL{best_strat}'][i+1:i+win+1]
            df['best'][i+1:i+win+1] = best_strat
            i+=win
        return df['final_sig'],df['final_trade'],df['best'],df['ch'],df['TP'],df['SL']
    
    master = pd.DataFrame()
    master['signals'],master['trade_type'],master['best'],master['ch'],master['TP'],master['SL'] = master_strategy(final,7,2)
    master.to_csv(r'C:\Users\harsh\Desktop\inter-iit\master_eth.csv')
    eth_4h = pd.read_csv(r'C:\Users\harsh\Desktop\inter-iit\eth_4h.csv')
    eth_4h.index = eth_4h['datetime']
    eth_4h = eth_4h['2020-01-01':]
    eth_4h['signals'] = master['signals'].astype(int)
    sig_to_trade(eth_4h)
    columns_to_keep = ['datetime','open', 'high', 'low', 'close', 'volume', 'signals', 'trade_type']
    eth_4h = eth_4h[columns_to_keep]
    eth_4h.drop('datetime',axis = 1,inplace = True)
    eth_4h.to_csv(r'C:\Users\harsh\Desktop\inter-iit\eth_final.csv')
        





def perform_backtest(csv_file_path):
    client = Client()
    result = client.backtest(
        jupyter_id="pj",  
        file_path=csv_file_path,
        leverage=1, 
        result_type = 'Q'
    )
    print(f"Backtest: ")
    for item in result:
        print(item)



if __name__ == "__main__":

    strat1,strat2,strat3,strat4 = process_data(r'C:\Users\harsh\Desktop\inter-iit\eth_4h.csv',r'C:\Users\harsh\Desktop\inter-iit\eth_1D.csv')
    strat(strat1,strat2,strat3,strat4)
    perform_backtest(r'C:\Users\harsh\Desktop\inter-iit\eth_final.csv')