import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import talib as ta
import seaborn as sns
from datetime import date
import warnings
import argparse
import sys
from pathlib import Path
import pickle

from prefect import flow, task, serve

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score,
)
import xgboost as xgb

# Add parent directory of 'src' to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from data_extraction import (
    extract_data,
    extract_date_feat,
    get_growth_df,
    get_future_growth_5d_df,
    relative_strength_index,
    sma_df,
    volatility,
    target_volatility,
    clean_dataframe_from_inf_and_nan,
)
from predictions.config import config


# ----------------------------- TASKS 1 -----------------------------

@task(retries=4, retry_delay_seconds=5)
def full_features(ticker: str) -> pd.DataFrame:
    """Generate full feature set for the given ticker."""
    df = extract_data(ticker)
    df = extract_date_feat(df, ticker)
    df = get_growth_df(df, ticker)
    df = get_future_growth_5d_df(df, ticker)
    df = sma_df(df, ticker)
    df = relative_strength_index(df, ticker)
    df = volatility(df, ticker)
    df = target_volatility(df, ticker)
    df = clean_dataframe_from_inf_and_nan(df)
    return df


# ----------------------------- TASKS 2 -----------------------------

@flow(name="plot_features_flow")
def plot_features(ticker:str, save_path: str = None):
    """Plot SMA 10, SMA 20, Close Price, and RSI."""
    print(f"📈 Generating feature plots for {ticker}...")

    dfm = full_features(ticker)

    keys = config.get_feature_keys(dfm)

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Plot close and moving averages
    sns.lineplot(data=dfm, x=dfm.index, y=f"{ticker.lower()}_close", label="Close", ax=ax1)
    sns.lineplot(data=dfm, x=dfm.index, y=f"{ticker.lower()}_sma10", label="SMA10", ax=ax1)
    sns.lineplot(data=dfm, x=dfm.index, y=f"{ticker.lower()}_sma20", label="SMA20", ax=ax1)

    ax1.set_title(f"{ticker} Close Price with SMA10 & SMA20")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(" ")


    # RSI subplot
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=dfm, x=dfm.index, y="rsi", ax=ax2, color="orange")
    ax2.set_title("RSI (Relative Strength Index)")
    ax2.axhline(70, color="red", linestyle="--", label="Overbought")
    ax2.axhline(30, color="green", linestyle="--", label="Oversold")
    ax2.legend()

    # Plot Hostorical Growth
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=dfm, x=dfm.index, y=f"growth_{ticker.lower()}_1d", label="Growth_1d", ax=ax3)
    sns.lineplot(data=dfm, x=dfm.index, y=f"growth_{ticker.lower()}_3d", label="Growth_3d", ax=ax3)
    sns.lineplot(data=dfm, x=dfm.index, y=f"growth_{ticker.lower()}_7d", label="Growth_7d", ax=ax3)
    sns.lineplot(data=dfm, x=dfm.index, y=f"growth_{ticker.lower()}_30d", label="Growth_30d", ax=ax3)
    # sns.lineplot(data=dfm, x=dfm.index, y=f"growth_{ticker.lower()}_90d", label="Growth_90d", ax=ax3)
    # sns.lineplot(data=dfm, x=dfm.index, y=f"growth_{ticker.lower()}_365d", label="Growth_365d", ax=ax3)

    ax3.set_title(f"{ticker} Historical Growth (1d/3d/7d/30d)")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Growth_Change")

    # Plot Hostorical Growth
    fig4, ax4 = plt.subplots(figsize=(12, 3))
    close = f"{ticker.lower()}_close"
    sns.scatterplot(x=dfm[close], y=dfm.target_volatility, ax=ax4)

    ax4.set_title(f"{ticker} - Closing Price vs Volatility")
    ax4.set_xlabel("Closing Price")
    ax4.set_ylabel("Volatility")

    # Save plot
    save_path = Path(__file__).resolve().parent.parent
    save_dir = Path(save_path).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_dir = save_dir / "plot_images"
    plot_path1 = out_dir / f"{ticker}_SMA.png"
    plot_path2 = out_dir / f"{ticker}_RSI.png"
    plot_path3 = out_dir / f"{ticker}_HG.png"
    plot_path4 = out_dir / f"{ticker}_Volatility.png"

    fig1.savefig(plot_path1, bbox_inches="tight")
    fig2.savefig(plot_path2, bbox_inches="tight")
    fig3.savefig(plot_path3, bbox_inches="tight")
    fig4.savefig(plot_path4, bbox_inches="tight")
    plt.close("all")   
    print(f"✅ Plot saved to {out_dir}")
    return str(out_dir)


# ----------------------------- FLOWS -----------------------------

@flow(name="feature_training_flow")
def feat(ticker: str, save_path: str = None):
    """Main pipeline for feature generation and model training."""
    print(f"\n🔹 Running financial prediction pipeline for: {ticker}\n")

    dfm = full_features(ticker)

    keys = config.get_feature_keys(dfm)

    # --- 5-DAY FUTURE GROWTH ---
    X = dfm[keys['X_5_Feat']]
    y = dfm["is_positive_growth_5d_future"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("5-DAY FUTURE GROWTH PREDICTION")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print("-" * 50)

    # --- MOVING AVERAGE ---
    X = dfm[keys['X_MA_Feat']]
    y = dfm["growing_moving_average"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("MOVING AVERAGE PREDICTION")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    print("-" * 50)

    # --- VOLATILITY ---
    X = dfm[keys['X_MA_Feat']]
    y = dfm["target_volatility"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    model_xgb = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost RMSE: {rmse:.5f}, R²: {r2:.3f}")
    print("=" * 50)

    # --- SAVE FILES ---
    # if save_path:
    #     save_dir = Path(save_path)
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     out_file = save_dir / f"{ticker}_features.csv"
    #     dfm.to_csv(out_file, index=True)
    #     print(f"✅ Features saved to {out_file}")
    if save_path:
        save_path = Path(__file__).resolve().parent.parent
        save_dir = Path(save_path).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / "dataset" / f"{ticker}_features.csv"
        dfm.to_csv(out_file, index=True)
        print(f"✅ Features saved to {out_file}")
    
    save_path = Path(__file__).resolve().parent.parent
    model_path = Path(save_path) / "dataset" / "stocks.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump((model, model_xgb), f)

    print(f"Models saved to: {save_path} folder")


@flow(name="prediction_flow")
def main(ticker: str, save_path: str = None):
    """Prediction flow: load trained model and predict latest 5-day volatility."""
    print(f"\n🔹 Running prediction flow for: {ticker}\n")
    dfm = full_features(ticker)
    keys = config.get_feature_keys(dfm)

    save_path = Path(__file__).resolve().parent.parent
    save_dir = Path(save_path).resolve()
    out_model = save_dir / "dataset" / "stocks.pkl"

    # print(out_model)

    with open(out_model, 'rb') as f:
        model, model_xgb = pickle.load(f)

    # Select features only, ignoring any NaN rows (from shift)
    valid_df = dfm[keys['X_MA_Feat']].dropna()

    latest_data = valid_df.iloc[[-1]]
    predicted_volatility = model_xgb.predict(latest_data)[0]
    print(latest_data)
    print(predicted_volatility)

    print(f"Predicted next 5-day volatility for {ticker}: {predicted_volatility*100:.2f}%")
    print("✅ Models successfully loaded!")


# ----------------------------- ENTRYPOINT -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Financial prediction pipeline using Prefect flows."
    )
    parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker symbol (e.g., AAPL, MSFT)"
    )
    parser.add_argument(
        "--save-path", type=str, default="./dataset", help="Optional path to save processed features"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["feat", "main", "plot_features", "serve"],
        default="feat",
        help="Choose whether to run 'feat', 'main', 'plot_features' or 'serve' both flows.",
    )

    args = parser.parse_args()

    if args.mode == "feat":
        feat(args.ticker.upper(), args.save_path)
    elif args.mode == "main":
        main(args.ticker.upper(), args.save_path)
    elif args.mode == "plot_features":
        plot_features(args.ticker.upper(), args.save_path)
    elif args.mode == "serve":
        serve(
            plot_features.with_options("Plot_Features_Flow"),
            feat.with_options(name="Feature_Training_Flow"),
            main.with_options(name="Prediction_Flow"),
        )
