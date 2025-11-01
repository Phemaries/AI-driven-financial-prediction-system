
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

import talib as ta
import seaborn as sns
from datetime import date
import warnings
warnings.filterwarnings('ignore')

# classification 
from sklearn import preprocessing

import xgboost as xgb
from sklearn.model_selection import train_test_split

import os

# metrics 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error, r2_score


min_date = "2000-01-01"
today = date.today()

# extract data 
def extract_data(ticker:str)->pd.DataFrame:
    df = f'df_{ticker}'
    df = yf.download(tickers = ticker, start=min_date, end=today, group_by=ticker)
    df.columns = df.columns.map('_'.join)
    df.columns = df.columns.str.lower()
    return df


# extract date features
def extract_date_feat(df:pd.DataFrame, prefix:str)->pd.DataFrame:
    prefix = prefix.lower()
    df[prefix+'_year'] = df.index.year
    df[prefix+'_month'] = df.index.month
    df[prefix+'_weekday'] = df.index.weekday
    df[prefix+'_date'] = df.index.date
    return df


# historical returns 
def get_growth_df(df:pd.DataFrame, prefix:str)->pd.DataFrame:
  prefix = prefix.lower()
  for i in [1,3,7,30,90,365]:
    df['growth_'+prefix+'_'+str(i)+'d'] = df[prefix+'_close'] / df[prefix+'_close'].shift(i)
    # GROWTH_KEYS = [k for k in df.keys() if k.startswith('growth')]
  return df


# value to predict 5-day future growth
def get_future_growth_5d_df(df:pd.DataFrame, prefix:str)->pd.DataFrame:
    prefix = prefix.lower()
    df[prefix+'_growth_future_5d'] = df[prefix+'_close'].shift(5) / df[prefix+'_close']
    # what we want to predict
    df['is_positive_growth_5d_future'] = np.where(df[prefix+'_growth_future_5d'] > 1, 1, 0)
    return df

# calculate relative strength index RSI
def relative_strength_index(df:pd.DataFrame, prefix:str)->pd.DataFrame:
    prefix = prefix.lower()
    df['rsi'] = ta.RSI(df[prefix+'_close'], timeperiod=14)
    return df


# Technical indicators
def sma_df(df:pd.DataFrame, prefix:str)->pd.DataFrame:
    prefix = prefix.lower()
    df[prefix+'_sma10']= df[prefix+'_close'].rolling(10).mean() # SimpleMovingAverage 10 days
    df[prefix+'_sma20']= df[prefix+'_close'].rolling(20).mean() # SimpleMovingAverage 20 days
    # what we want to predict (moving average)
    df['growing_moving_average'] = np.where(df[prefix+'_sma10'] > df[prefix+'_sma20'], 1, 0)
    return df      


# current volatility 30d
def volatility(df:pd.DataFrame, prefix:str)->pd.DataFrame:
    prefix = prefix.lower()
    returns = df[prefix+'_close'].pct_change()
    df['volatility_30d'] =   returns.rolling(30).std().shift(-30) * np.sqrt(252)
    return df


# predict future volatility i.e. next 30-day volatility
def target_volatility(df:pd.DataFrame, prefix:str)->pd.DataFrame:
    prefix = prefix.lower()
    returns = df[prefix+'_close'].pct_change()
    df['target_volatility'] =   returns.rolling(30).std().shift(-30) * np.sqrt(252)
    return df

# +-inf to NaN, all NaNs to 0s
def clean_dataframe_from_inf_and_nan(df:pd.DataFrame)->pd.DataFrame:
  df.replace([np.inf, -np.inf], np.nan, inplace=True)
  df.fillna(0, inplace=True)
  return df


# def full_features():
#     df = extract_data('MSFT')
#     df = extract_date_feat(df, 'MSFT')
#     df = get_growth_df(df, 'MSFT')
#     df = get_future_growth_5d_df(df, 'MSFT')
#     df = sma_df(df, 'MSFT')
#     df = relative_strength_index(df, 'MSFT')
#     df = volatility(df, 'MSFT')
#     df = target_volatility(df, 'MSFT')
#     df = clean_dataframe_from_inf_and_nan(df)
#     dfm = df.copy()
#     return dfm







