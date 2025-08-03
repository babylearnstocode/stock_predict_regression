import joblib
import pandas as pd
from datetime import timedelta
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

# Load your trained model and scaler
svr_best = joblib.load("svr_best.pkl")
scaler = joblib.load("scaler.pkl")

# Load your data

index_ticker = "^GSPC"  # S&P 500 index
etf_ticker = "SP500S.SW"  # UBS ETF (CHF hedged)

start_date = "1972-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")


print("Fetching S&P 500 index data...")
index_data = yf.download(index_ticker, start=start_date, end=end_date)

n_days = 252  # Number of trading days in a year
future_dates = []
future_predictions = []


def add_original_features(df, df_new):
    df_new["open"] = df["Open"]
    df_new["open_1"] = df["Open"].shift(1)
    df_new["close_1"] = df["Close"].shift(1)
    df_new["high_1"] = df["High"].shift(1)
    df_new["low_1"] = df["Low"].shift(1)
    df_new["volume_1"] = df["Volume"].shift(1)
    return df_new


def add_avg_price(df, df_new):
    df_new["avg_price_5"] = df["Close"].rolling(5).mean().shift(1)
    df_new["avg_price_30"] = df["Close"].rolling(30).mean().shift(1)
    df_new["avg_price_365"] = df["Close"].rolling(365).mean().shift(1)
    df_new["ratio_avg_price_5_30"] = df_new["avg_price_5"] / df_new["avg_price_30"]
    df_new["ratio_avg_price_30_365"] = df_new["avg_price_30"] / df_new["avg_price_365"]
    df_new["ratio_avg_price_5_365"] = df_new["avg_price_5"] / df_new["avg_price_365"]
    return df_new


def add_avg_volume(df, df_new):
    df_new["avg_volume_5"] = df["Volume"].rolling(5).mean().shift(1)
    df_new["avg_volume_30"] = df["Volume"].rolling(30).mean().shift(1)
    df_new["avg_volume_365"] = df["Volume"].rolling(365).mean().shift(1)
    df_new["ratio_avg_volume_5_30"] = df_new["avg_volume_5"] / df_new["avg_volume_30"]
    df_new["ratio_avg_volume_30_365"] = (
        df_new["avg_volume_30"] / df_new["avg_volume_365"]
    )
    df_new["ratio_avg_volume_5_365"] = df_new["avg_volume_5"] / df_new["avg_volume_365"]
    return df_new


def add_std_price(df, df_new):
    df_new["std_price_5"] = df["Close"].rolling(5).std().shift(1)
    df_new["std_price_30"] = df["Close"].rolling(30).std().shift(1)
    df_new["std_price_365"] = df["Close"].rolling(365).std().shift(1)
    df_new["ratio_std_price_5_30"] = df_new["std_price_5"] / df_new["std_price_30"]
    df_new["ratio_std_price_30_365"] = df_new["std_price_30"] / df_new["std_price_365"]
    df_new["ratio_std_price_5_365"] = df_new["std_price_5"] / df_new["std_price_365"]
    return df_new


def add_std_volume(df, df_new):
    df_new["std_volume_5"] = df["Volume"].rolling(5).std().shift(1)
    df_new["std_volume_30"] = df["Volume"].rolling(30).std().shift(1)
    df_new["std_volume_365"] = df["Volume"].rolling(365).std().shift(1)
    df_new["ratio_std_volume_5_30"] = df_new["std_volume_5"] / df_new["std_volume_30"]
    df_new["ratio_std_volume_30_365"] = (
        df_new["std_volume_30"] / df_new["std_volume_365"]
    )
    df_new["ratio_std_volume_5_365"] = df_new["std_volume_5"] / df_new["std_volume_365"]
    return df_new


def add_return_features(df, df_new):
    df_new["return_1"] = (
        (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    ).shift(1)
    df_new["return_5"] = (
        (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5)
    ).shift(1)
    df_new["return_30"] = (
        (df["Close"] - df["Close"].shift(21)) / df["Close"].shift(21)
    ).shift(1)
    df_new["return_365"] = (
        (df["Close"] - df["Close"].shift(252)) / df["Close"].shift(252)
    ).shift(1)
    df_new["moving_avg_5"] = df["Close"].rolling(5).mean().shift(1)
    df_new["moving_avg_30"] = df["Close"].rolling(21).mean().shift(1)
    df_new["moving_avg_365"] = df["Close"].rolling(252).mean().shift(1)

    return df_new


def generate_features(df):
    df_new = pd.DataFrame()
    df_new = add_original_features(df, df_new)
    df_new = add_avg_price(df, df_new)
    df_new = add_avg_volume(df, df_new)
    df_new = add_std_price(df, df_new)
    df_new = add_std_volume(df, df_new)
    df_new = add_return_features(df, df_new)

    # the target
    df_new["close"] = df["Close"]
    df_new = df_new.dropna(axis=0)
    return df_new


latest_data = index_data.copy()
features_df = generate_features(latest_data)
for i in range(n_days):
    # Generate features for the latest data
    features_df = generate_features(latest_data)
    X_future = features_df.drop(columns=["close"], axis=1).values[-1:]
    X_future_scaled = scaler.transform(X_future)

    # Predict next close price
    next_close = svr_best.predict(X_future_scaled)[0]
    future_predictions.append(next_close)

    # Get the next date (business day)
    last_date = latest_data.index[-1]
    next_date = last_date + timedelta(days=1)
    # Ensure next_date is a business day (skip weekends)
    while next_date.weekday() > 4:  # 0=Monday, 4=Friday
        next_date += timedelta(days=1)
    future_dates.append(next_date)

    # Create a new row for the next day
    new_row = latest_data.iloc[-1].copy()
    new_row["close"] = next_close
    new_row["open_1"] = next_close
    new_row["high_1"] = next_close
    new_row["low_1"] = next_close
    new_row["volume_1"] = latest_data["Volume"].mean()
    new_row.name = next_date
    latest_data.loc[next_date] = new_row


fig = go.Figure()
fig.add_trace(go.Scatter(x=features_df.index, y=features_df["close"], name="Actual"))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, name="SVR"))
fig.update_layout(
    title="S&P 500 Index",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    template="plotly_white",
)
fig.show()
