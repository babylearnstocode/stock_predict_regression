import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import joblib
from datetime import datetime

scaler = StandardScaler()

# Define tickers and date range
index_ticker = "^GSPC"  # S&P 500 index
etf_ticker = "SP500S.SW"  # UBS ETF (CHF hedged)

start_date = "1972-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")


print("Fetching S&P 500 index data...")
index_data = yf.download(index_ticker, start=start_date, end=end_date)


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


data = generate_features(index_data)

start_train = "1972-01-01"
end_train = "2023-12-31"

start_test = "2024-01-01"
end_test = "2025-07-31"

train_data = data[start_train:end_train]
X_train = train_data.drop(columns=["close"], axis=1).values
Y_train = train_data["close"].values

test_data = data[start_test:end_test]
X_test = test_data.drop(columns=["close"], axis=1).values
Y_test = test_data["close"].values


X_scaled_train = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl")
X_scaled_test = scaler.transform(X_test)


# SGD Regressor
# param_grid = {"alpha": [1e-4, 3e-4, 1e-3], "eta0": [0.01, 0.03, 0.1]}
# sgd_regressor = SGDRegressor(penalty="l2", max_iter=1000, random_state=42)
# grid_search = GridSearchCV(sgd_regressor, param_grid, cv=5, scoring="r2")
# grid_search.fit(X_scaled_train, Y_train)
# sgd_best = grid_search.best_estimator_
# predictions_sgd = sgd_best.predict(X_scaled_test)
# mse_sgd = mean_squared_error(Y_test, predictions_sgd)
# mae_sgd = mean_absolute_error(Y_test, predictions_sgd)
# r2_sgd = r2_score(Y_test, predictions_sgd)
# joblib.dump(sgd_best, "sgd_best.pkl")
# print(f"SGD Regressor - MSE: {mse_sgd}, MAE: {mae_sgd}, R2: {r2_sgd}")

# param_grid = {
#     "max_depth": [30, 50],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [3, 5],
# }
# rf_regressor = RandomForestRegressor(
#     random_state=42, n_estimators=100, n_jobs=-1, max_features="sqrt"
# )
# grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring="r2", n_jobs=-1)
# grid_search.fit(X_train, Y_train)
# rf_best = grid_search.best_estimator_
# predictions_rf = rf_best.predict(X_test)
# mse_rf = mean_squared_error(Y_test, predictions_rf)
# mae_rf = mean_absolute_error(Y_test, predictions_rf)
# r2_rf = r2_score(Y_test, predictions_rf)
# joblib.dump(rf_best, "rf_best.pkl")
# print(f"Random Forest Regressor - MSE: {mse_rf}, MAE: {mae_rf}, R2: {r2_rf}")

param_grid = [
    {
        "C": [100, 300, 500],
        "kernel": ["linear"],
        "epsilon": [0.00003, 0.0001],
    },
    {
        "C": [10, 100, 1000],
        "kernel": ["rbf"],
        "epsilon": [0.00003, 0.0001],
    },
]
svr_regressor = SVR()
grid_search = GridSearchCV(svr_regressor, param_grid, cv=5, scoring="r2", n_jobs=-1)
grid_search.fit(X_scaled_train, Y_train)
svr_best = grid_search.best_estimator_
predictions_svr = svr_best.predict(X_scaled_test)
mse_svr = mean_squared_error(Y_test, predictions_svr)
mae_svr = mean_absolute_error(Y_test, predictions_svr)
r2_svr = r2_score(Y_test, predictions_svr)
joblib.dump(svr_best, "svr_best.pkl")
print(f"SVR Regressor - MSE: {mse_svr}, MAE: {mae_svr}, R2: {r2_svr}")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data["close"], name="Actual"))
# fig.add_trace(go.Scatter(x=test_data.index, y=predictions_sgd, name="SGD"))
# fig.add_trace(go.Scatter(x=test_data.index, y=predictions_rf, name="RF"))
fig.add_trace(go.Scatter(x=test_data.index, y=predictions_svr, name="SVR"))
fig.update_layout(
    title="S&P 500 Index",
    xaxis_title="Date",
    yaxis_title="Price",
    hovermode="x unified",
    template="plotly_white",
)
fig.show()

# fig.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close"))
# fig.add_trace(go.Scatter(x=data.index, y=data["moving_avg_5"], name="Moving Avg 5"))
# fig.add_trace(go.Scatter(x=data.index, y=data["moving_avg_30"], name="Moving Avg 30"))
# fig.add_trace(go.Scatter(x=data.index, y=data["moving_avg_365"], name="Moving Avg 365"))
# fig.update_layout(
#     title="S&P 500 Index",
#     xaxis_title="Date",
#     yaxis_title="Price",
#     hovermode="x unified",
#     template="plotly_white",
# )
# fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
# fig.show()
