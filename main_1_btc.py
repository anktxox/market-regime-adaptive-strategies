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
#os.chdir(r'C:\Users\Ankit\Desktop\inter-iit\untrade-sdk')
#untrade1= os.path.abspath(r'C:\Users\Ankit\Desktop\inter-iit\untrade-sdk')
sys.path.append(untrade1)
from untrade.client import Client
client = Client()
import pandas_ta as ta
from filterpy.kalman import KalmanFilter




def process_data(filepath_4h,filepath_1D):


    def process_btc1_data(csv_file_path):
        df = pd.read_csv(csv_file_path)

        # Calculate Technical Indicators
        df['CHOP'] = ta.chop(high=df['high'], low=df['low'], close=df['close'], length=14, append=True)
        df.ta.aroon(high='high', low='low', length=14, append=True)

        # Define the strategy function
        def strat(data):
            # Load and clean data
            data['datetime'] = pd.to_datetime(data['datetime'])
            data = data[data['datetime'] >= '2020-01-01 00:00:00']
            data = data[data['volume'] != 0].reset_index(drop=True)

            data['signals'] = 0  # Signals column
            position = 0  # Current position: 1 for long, -1 for short, 0 for no position
            SR = None  # Support/Resistance level for current position

            for i in range(3, len(data)):
                # Signal generation
                sell_condition = all([
                    data['low'].iloc[i] < data['low'].iloc[i - 1],
                    data['low'].iloc[i - 1] < data['high'].iloc[i],
                    data['high'].iloc[i] < data['low'].iloc[i - 2],
                    data['low'].iloc[i - 2] < data['high'].iloc[i - 1],
                    data['high'].iloc[i - 1] < data['low'].iloc[i - 3],
                    data['low'].iloc[i - 3] < data['high'].iloc[i - 2],
                    data['high'].iloc[i - 2] < data['high'].iloc[i - 3],
                ])

                buy_condition = all([
                    data['high'].iloc[i] > data['high'].iloc[i - 1],
                    data['high'].iloc[i - 1] > data['low'].iloc[i],
                    data['low'].iloc[i] > data['high'].iloc[i - 2],
                    data['high'].iloc[i - 2] > data['low'].iloc[i - 1],
                    data['low'].iloc[i - 1] > data['high'].iloc[i - 3],
                    data['high'].iloc[i - 3] > data['low'].iloc[i - 2],
                    data['low'].iloc[i - 2] > data['low'].iloc[i - 3],
                ])

                if sell_condition and data['AROONU_14'].iloc[i] <= data['AROOND_14'].iloc[i]:
                        data.loc[i, 'signals'] = -1  # Open short
                        if position == 0:
                            position = -1
                            SR = data['high'].iloc[i]  # Set resistance level for short

                elif buy_condition and data['AROONU_14'].iloc[i] >= data['AROOND_14'].iloc[i]:
                        data.loc[i, 'signals'] = 1  # Open long
                        if position == 0:
                            position = 1
                            SR = data['low'].iloc[i]  # Set support level for long

                elif position == 1:  # Manage long position
                    if data['low'].iloc[i] <= SR:
                        data.loc[i, 'signals'] = -1  # Close long
                        position = 0
                        SR = None  # Reset SR

                elif position == -1:  # Manage short position
                    if data['high'].iloc[i] >= SR:  # Exit short if price breaches resistance
                        data.loc[i, 'signals'] = 1  # Close short
                        position = 0
                        SR = None  # Reset SR

            return data

        # Run the strategy
        test = strat(df)

        # Additional indicators and filtering
        test['CHOP'] = ta.chop(high=test['high'], low=test['low'], close=test['close'], length=14)
        test['Choppy_Market'] = test['CHOP'] > 60
        test.loc[test['Choppy_Market'] == True, 'signals'] = 0

        test['Rolling_Std'] = test['close'].rolling(window=20).std()
        test['Normalized_Std'] = test['Rolling_Std'] / test['close']

        threshold = 0.010
        test['Stagnant_Market'] = test['Normalized_Std'] < threshold

        test['Stagnant_Market'] = test['Stagnant_Market'].fillna(False)
        test.loc[test['Stagnant_Market'] == True, 'signals'] = 0

        final_data = test.copy()

        # Stop Loss Function
        def add_stop_loss(data):
            data['SL'] = 0.0  # Initialize the 'SL' column
            for index, row in data.iterrows():
                if row['signals'] == 1:
                    data.loc[index, 'SL'] = row['close'] * 0.8
                elif row['signals'] == -1:
                    data.loc[index, 'SL'] = row['close'] * 1.2
            return data

        final_data = add_stop_loss(final_data)

        # Take Profit Function
        def add_take_profit(data):
            data['TP'] = 0.0  # Initialize the 'TP' column
            for index, row in data.iterrows():
                if row['signals'] == 1:
                    data.loc[index, 'TP'] = row['close'] * 1.4
                elif row['signals'] == -1:
                    data.loc[index, 'TP'] = row['close'] * 0.6
            return data

        final_data = add_take_profit(final_data)

        data = final_data[['close', 'datetime', 'TP', 'SL', 'signals']]

        # Trade logic to determine long/short positions
        trade_s = False  # short position flag
        trade_l = False  # long position flag
        data = data.copy()
        data['trade'] = ''  # initialize trade column with empty string

        for index, row in data.iterrows():
            if row['signals'] == 1:  # long signal
                if trade_s == False and trade_l == False:
                    data.at[index, 'trade'] = 'long'
                    trade_l = True  # set long flag
                elif trade_s == True and trade_l == False:
                    data.at[index, 'trade'] = 'no_trade'  # exit short, enter long
                    trade_s = False  # reset short flag
                elif trade_s == False and trade_l == True:
                    data.at[index, 'trade'] = 'long'  # already in long, maintain position

            elif row['signals'] == -1:  # short signal
                if trade_s == False and trade_l == False:
                    data.at[index, 'trade'] = 'short'
                    trade_s = True  # set short flag
                elif trade_s == False and trade_l == True:
                    data.at[index, 'trade'] = 'no_trade'  # exit long, enter short
                    trade_l = False  # reset long flag
                elif trade_s == True and trade_l == False:
                    data.at[index, 'trade'] = 'short'  # already in short, maintain position

            elif row['signals'] == 2:  # close long
                if trade_l == True:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_l = False  # reset long flag

            elif row['signals'] == -2:  # close short
                if trade_s == True:
                    data.at[index, 'trade'] = 'no_trade'
                    trade_s = False  # reset short flag

            elif row['signals'] == 0:  # no trade signal
                if trade_s == True:
                    data.at[index, 'trade'] = 'short'  # stay in short position
                elif trade_l == True:
                    data.at[index, 'trade'] = 'long'  # stay in long position
                else:
                    data.at[index, 'trade'] = 'no_trade'  # not in trade
        data['trade_type'] = data['trade']
        data = data[['datetime', 'close', 'signals', 'trade_type']]
        return data
    
    def process_btc2_data(data):
        def preprocessing(df):
            df = df.copy()
            rsi = ta.rsi(df["close"], length=12)
            ema_9 = ta.ema(df["close"], length=9)
            ema_20 = ta.ema(df["close"], length=20)
            sar = ta.psar(high=df['high'], low=df['low'], close=df['close'], af=0.02, max_af=0.2)
            df['RSI'] = rsi
            df['EMA_9'] = ema_9
            df['EMA_20'] = ema_20
            df['SAR'] = sar['PSARr_0.02_0.2']
            ema_6_rsi = ta.ema(df["RSI"], timeperiod=6)
            df['EMA_6_RSI'] = ema_6_rsi
            df['p_SAR'] = df['SAR'].shift(1)
            df['p_RSI'] = df['RSI'].shift(2)
            df = df.dropna()
            return df

        def signal_gen(df):
            df['signals'] = 0
            df['trade_type'] = 'No_Trade'
            trade_l = False
            trade_s = False

            for index, row in df.iterrows():
                if row['EMA_6_RSI'] > 60:
                    if row['EMA_9'] > row['EMA_20']:
                        if trade_s == False and trade_l == True:
                            df.at[index, 'signals'] = 0
                            df.at[index, 'trade_type'] = 'hold'
                        elif trade_s == True and trade_l == False:
                            df.at[index, 'signals'] = 2
                            df.at[index, 'trade_type'] = 'long_reversal'
                            trade_s = False
                            trade_l = True
                        elif trade_s == False and trade_l == False:
                            df.at[index, 'signals'] = 1
                            df.at[index, 'trade_type'] = 'long'
                            trade_s = False
                            trade_l = True
                    else:
                        if trade_s == True and trade_l == False:
                            df.at[index, 'signals'] = 0
                            df.at[index, 'trade_type'] = 'hold'
                        elif trade_s == False and trade_l == True:
                            df.at[index, 'signals'] = -2
                            df.at[index, 'trade_type'] = 'short_reversal'
                            trade_s = True
                            trade_l = False
                        elif trade_s == False and trade_l == False:
                            df.at[index, 'signals'] = -1
                            df.at[index, 'trade_type'] = 'short'
                            trade_s = True
                            trade_l = False
                else:
                    if row["EMA_6_RSI"] < 40:
                        if (row["SAR"] <= row["close"]) and (row["p_SAR"] > row["SAR"]):
                            df.at[index, 'signals'] = 0
                            df.at[index, 'trade_type'] = 'hold'
                        else:
                            if trade_s == False and trade_l == True:
                                df.at[index, 'signals'] = -1
                                df.at[index, 'trade_type'] = 'long_exit'
                                trade_s = False
                                trade_l = False
                    else:
                        if row['RSI'] > 55 and row["p_RSI"] > row["RSI"]:
                            if trade_s == False and trade_l == True:
                                df.at[index, 'signals'] = -1
                                df.at[index, 'trade_type'] = 'long_exit'
                                trade_s = False
                                trade_l = False
                        else:
                            df.at[index, 'signals'] = 0
                            df.at[index, 'trade_type'] = 'hold'
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

        processed_data = preprocessing(data)
        strategy_data = signal_gen(processed_data)
        final_data = trade_gen(strategy_data)
        return final_data[['datetime', 'close', 'signals', 'trade', 'trade_type']]


    def process_btc3_data(filepath):
        df = pd.read_csv(filepath)

        # Calculate indicators
        df['CHOP'] = ta.chop(high=df['high'], low=df['low'], close=df['close'], length=14)
        df.ta.aroon(high='high', low='low', length=14, append=True)

        # Data preparation
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df[df['datetime'] >= '2020-01-01 00:00:00']
        df = df[df['volume'] != 0].reset_index(drop=True)
        df['signals'] = 0

        position = 0
        SR = None

        # Trading logic
        for i in range(3, len(df)):
            sell_condition = all([
                df['low'].iloc[i] < df['low'].iloc[i - 1],
                df['low'].iloc[i - 1] < df['high'].iloc[i],
                df['high'].iloc[i] < df['low'].iloc[i - 2],
                df['low'].iloc[i - 2] < df['high'].iloc[i - 1],
                df['high'].iloc[i - 1] < df['low'].iloc[i - 3],
                df['low'].iloc[i - 3] < df['high'].iloc[i - 2],
                df['high'].iloc[i - 2] < df['high'].iloc[i - 3],
            ])
            buy_condition = all([
                df['high'].iloc[i] > df['high'].iloc[i - 1],
                df['high'].iloc[i - 1] > df['low'].iloc[i],
                df['low'].iloc[i] > df['high'].iloc[i - 2],
                df['high'].iloc[i - 2] > df['low'].iloc[i - 1],
                df['low'].iloc[i - 1] > df['high'].iloc[i - 3],
                df['high'].iloc[i - 3] > df['low'].iloc[i - 2],
                df['low'].iloc[i - 2] > df['low'].iloc[i - 3],
            ])

            if sell_condition and df['CHOP'].iloc[i] < 38:
                df.loc[i, 'signals'] = -1
                if position == 0:
                    position = -1
                    SR = df['high'].iloc[i]
            elif buy_condition and df['CHOP'].iloc[i] < 38:
                df.loc[i, 'signals'] = 1
                if position == 0:
                    position = 1
                    SR = df['low'].iloc[i]
            elif position == 1 and df['low'].iloc[i] <= SR:
                df.loc[i, 'signals'] = -1
                position = 0
                SR = None
            elif position == -1 and df['high'].iloc[i] >= SR:
                df.loc[i, 'signals'] = 1
                position = 0
                SR = None

        # Add rolling standard deviation
        df['Rolling_Std'] = df['close'].rolling(window=20).std()
        df['Normalized_Std'] = df['Rolling_Std'] / df['close']
        df['Stagnant_Market'] = df['Normalized_Std'] < 0.010
        df['Stagnant_Market'] = df['Stagnant_Market'].fillna(False)
        df.loc[df['Stagnant_Market'] == True, 'signals'] = 0

        # Add stop-loss
        df['SL'] = df.apply(lambda row: row['close'] * 0.8 if row['signals'] == 1 else (
            row['close'] * 1.2 if row['signals'] == -1 else 0.0), axis=1)

        # Add take-profit
        df['TP'] = df.apply(lambda row: row['close'] * 1.4 if row['signals'] == 1 else (
            row['close'] * 0.6 if row['signals'] == -1 else 0.0), axis=1)

        # Add trade state
        trade_s = False  # short position flag
        trade_l = False  # long position flag
        df = df.copy()
        df['trade'] = ''  # initialize trade column with empty string

        for index, row in df.iterrows():
            if row['signals'] == 1:  # long signal
                if trade_s == False and trade_l == False:
                    df.at[index, 'trade'] = 'long'
                    trade_l = True  # set long flag
                elif trade_s == True and trade_l == False:
                    df.at[index, 'trade'] = 'no_trade'  # exit short, enter long
                    trade_s = False  # reset short flag
                elif trade_s == False and trade_l == True:
                    df.at[index, 'trade'] = 'long'  # already in long, maintain position

            elif row['signals'] == -1:  # short signal
                if trade_s == False and trade_l == False:
                    df.at[index, 'trade'] = 'short'
                    trade_s = True  # set short flag
                elif trade_s == False and trade_l == True:
                    df.at[index, 'trade'] = 'no_trade'  # exit long, enter short
                    trade_l = False  # reset long flag
                elif trade_s == True and trade_l == False:
                    df.at[index, 'trade'] = 'short'  # already in short, maintain position

            elif row['signals'] == 2:  # close long
                if trade_l == True:
                    df.at[index, 'trade'] = 'no_trade'
                    trade_l = False  # reset long flag

            elif row['signals'] == -2:  # close short
                if trade_s == True:
                    df.at[index, 'trade'] = 'no_trade'
                    trade_s = False  # reset short flag

            elif row['signals'] == 0:  # no trade signal
                if trade_s == True:
                    df.at[index, 'trade'] = 'short'  # stay in short position
                elif trade_l == True:
                    df.at[index, 'trade'] = 'long'  # stay in long position
                else:
                    df.at[index, 'trade'] = 'no_trade'  # not in trade

        # Final columns
        df['trade_type'] = df['trade']
        return df[['datetime', 'close', 'TP', 'SL', 'signals', 'trade_type']]


    def process_btc4_data(data):
        def btc_kalman(data,start_date='2020-01-01', sl_percent=5,tp_percent=25, volume_rolling_window=14, kf_R=100, kf_Q_diag=(0.001, 0.001),stc_threshold=50):
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
            def generate_signals(df,sl_percent,tp_percent,start_date, stc_threshold=50):
                current_position = None
                entry_price = None
                tp = None
                sl = None
                signals = [0] * len(df)
                start_idx = df.index.get_loc(start_date)
                rolling_volume = df['volume'].rolling(window=volume_rolling_window).mean()

                # Iterate over the DataFrame from the start index
                for i in range(start_idx, len(df)):
                    # Bollinger Band and ATR confirmation
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
                        if vol_condition and stc_bullish:
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
                        if vol_condition and stc_bearish:
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
                            df.at[index, 'trade_type'] = 'no_trade'
                return df


            df['smoothed_price'] = smoothed_prices
            df = generate_signals(df,sl_percent,tp_percent,start_date,stc_threshold)
            df = df['2020-01-01':]
            df = assign_trades(df)
            df['datetime'] = df.index
            df = df[['close','signals','trade_type','datetime']]
            return df

        start_date = '2020-01-01'
        sl_percent = 7
        tp_percent = 100
        volume_rolling_window = 14
        kf_R = 100
        kf_Q_diag = (0.001, 0.001)
        stc = 50

        data = btc_kalman(data,start_date=start_date,sl_percent=sl_percent,tp_percent=tp_percent,volume_rolling_window=volume_rolling_window,kf_R=kf_R, kf_Q_diag=kf_Q_diag,stc_threshold=stc)
        return data


    btc_1d = pd.read_csv(filepath_1D)
    strat1 = process_btc1_data(filepath_4h)
    strat2 = process_btc2_data(btc_1d)
    strat3 = process_btc3_data(filepath_4h)
    strat4 = process_btc4_data(btc_1d)

    return strat1,strat2,strat3,strat4



def strat(start1,strat2,strat3,strat4):
    strat1['datetime'] = pd.to_datetime(strat1['datetime'])
    strat1.index = strat1['datetime']
    strat2['datetime'] = pd.to_datetime(strat2['datetime'])
    new_index = pd.date_range(start=strat2['datetime'].min(), end=strat2['datetime'].max(), freq="4H")
    strat2 = strat2.set_index('datetime').reindex(new_index)
    strat2['close'] = strat1['close']
    strat2['signals'] = strat2['signals'].fillna(0)
    strat2['trade_type'] = strat2['trade_type'].fillna(method = 'ffill')
    strat3['datetime'] = pd.to_datetime(strat3['datetime'])
    strat3.index = strat3['datetime']
    strat4['datetime'] = pd.to_datetime(strat4['datetime'])
    new_index = pd.date_range(start=strat4['datetime'].min(), end=strat4['datetime'].max(), freq="4H")
    strat4 = strat4.set_index('datetime').reindex(new_index)
    strat4['close'] = strat1['close']
    strat4['signals'] = strat4['signals'].fillna(0)
    strat4['trade_type'] = strat4['trade_type'].fillna(method = 'ffill')

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
    final['booksize1'] = strat1['booksize']
    final['sig2'] = strat2['signals']
    final['trade_type2'] = strat2['trade_type']
    final['booksize2'] = strat2['booksize']
    final['sig3'] = strat3['signals']
    final['trade_type3'] = strat3['trade_type']
    final['booksize3'] = strat3['booksize']
    final['sig4'] = strat4['signals']
    final['trade_type4'] = strat4['trade_type']
    final['booksize4'] = strat4['booksize']


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
            weights['volatility'] * volatility2 +
            weights['mean_pos_ret'] * mean_pos_ret2 +
            weights['max_drawdown'] * max_drawdown2 +
            weights['sharpe_ratio'] * sharpe2 
        )

        rank[1] = (
            weights['volatility'] * volatility1 +
            weights['mean_pos_ret'] * mean_pos_ret1 +
            weights['max_drawdown'] * max_drawdown1 +
            weights['sharpe_ratio'] * sharpe1 
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
            df['best'][i+1:i+win+1] = best_strat
            i+=win
        return df['final_sig'],df['final_trade'],df['best'],df['ch']
    
    master = pd.DataFrame()
    master['signals'],master['trade_type'],master['best'],master['ch'] = master_strategy(final,150,2)
    master.to_csv(r'C:\Users\harsh\Desktop\inter-iit\master.csv')
    btc_4h = pd.read_csv(r'C:\Users\harsh\Desktop\inter-iit\BTC_2019_2023_4h.csv')
    btc_4h.index = btc_4h['datetime']
    btc_4h = btc_4h['2020-01-01':]
    master.index = btc_4h['datetime']
    btc_4h['signals'] = master['signals'].astype(int)
    sig_to_trade(btc_4h)
    columns_to_keep = ['datetime','open', 'high', 'low', 'close', 'volume', 'signals', 'trade_type']
    btc_4h = btc_4h[columns_to_keep]
    btc_4h.drop('datetime',axis = 1,inplace = True)
    btc_4h.to_csv(r'C:\Users\harsh\Desktop\inter-iit\btc_final.csv')
        



def perform_backtest(csv_file_path):
    client = Client()
    result = client.backtest(
        jupyter_id="pj",  # the one you use to login to jupyter.untrade.io
        file_path=csv_file_path,
        leverage=1,  # Adjust leverage as needed
        result_type = 'Q'
    )
    print(f"Backtest: ")
    for item in result:
        print(item)


if __name__ == "__main__":
    #Input 4h and 1H data frame
    strat1,strat2,strat3,strat4 = process_data(r'C:\Users\Ankit\Desktop\inter-iit\BTC_2019_2023_4h.csv',r'C:\Users\harsh\Desktop\inter-iit\btc_1D.csv')
    strat(strat1,strat2,strat3,strat4)
    perform_backtest(r'C:\Users\Ankit\Desktop\inter-iit\btc_final.csv')
