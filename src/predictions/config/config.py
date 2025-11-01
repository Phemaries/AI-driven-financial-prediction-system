# config.py

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'JPM', 'CVX', 'META', 'ORCL']

def get_feature_keys(dfm):
    """Generate column key groups from a given DataFrame."""
    DATE_KEYS = [k for k in dfm.keys() if k.endswith(('year', 'month', 'weekday', 'date'))]
    OHLCV_KEYS = [k for k in dfm.keys() if k.endswith(('open', 'high', 'low', 'close', 'volume'))]
    SMA_KEYS = [k for k in dfm.keys() if k.endswith(('sma10', 'sma20'))]
    GROWTH_KEYS = [k for k in dfm.keys() if k.startswith('growth')]
    X_5_Feat = [k for k in dfm.keys() if k.endswith(('close', 'volume', '7d'))] + ['rsi']
    X_MA_Feat = [k for k in dfm.keys() if k.endswith(('close', 'volume', '7d', 'sma20'))] + ['rsi']

    return {
        'DATE_KEYS': DATE_KEYS,
        'OHLCV_KEYS': OHLCV_KEYS,
        'SMA_KEYS': SMA_KEYS,
        'GROWTH_KEYS': GROWTH_KEYS,
        'X_5_Feat': X_5_Feat,
        'X_MA_Feat': X_MA_Feat,
    }


