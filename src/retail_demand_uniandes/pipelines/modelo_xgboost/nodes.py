import pandas as pd
import numpy as np
import holidays
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

def concat_dicttodf(datadict):
    return pd.concat(datadict.values(), keys=datadict.keys())

def create_intermediate_sales_table(df, category_col="category", date_col="date", target_col="units_sold"):
    df[date_col] = pd.to_datetime(df[date_col])
    categories = df[category_col].unique()
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    full_days = pd.date_range(min_date, max_date, freq='D')
    full_index = pd.DataFrame([(cat, d) for cat in categories for d in full_days], columns=[category_col, date_col])
    agg_df = df.groupby([category_col, date_col])[target_col].sum().reset_index()
    intermediate = full_index.merge(agg_df, on=[category_col, date_col], how="left").fillna({target_col: 0})
    return intermediate

def add_brazil_events(df, date_col="date"):
    years = df[date_col].dt.year.unique()
    br_holidays = holidays.Brazil(years=years)
    df["is_holiday"] = df[date_col].isin(br_holidays).astype(int)
    df["is_blackfriday"] = 0
    for year in years:
        nov = df[df[date_col].dt.year == year]
        fridays = nov[nov[date_col].dt.month == 11]
        fridays = fridays[fridays[date_col].dt.weekday == 4]
        if not fridays.empty:
            last_friday = fridays[date_col].max()
            df.loc[df[date_col] == last_friday, "is_blackfriday"] = 1
    return df

def preprocess_and_split(df, category_col, date_col, target_col, test_days):
    intermediate = create_intermediate_sales_table(df, category_col, date_col, target_col)
    categories = intermediate[category_col].unique()
    traindict, testdict = {}, {}
    for cat in categories:
        df_cat = intermediate[intermediate[category_col] == cat].copy()
        df_cat = add_brazil_events(df_cat, date_col)  # << Añadir eventos especiales aquí
        max_date = df_cat[date_col].max()
        split_date = max_date - pd.Timedelta(days=test_days)
        train = df_cat[df_cat[date_col] < split_date].copy()
        test = df_cat[df_cat[date_col] >= split_date].copy()
        traindict[cat] = train
        testdict[cat] = test
    return traindict, testdict, intermediate

def feature_engineering(df):
    df = df.copy()
    df["lag_1"] = df["units_sold"].shift(1)
    df["roll_7"] = df["units_sold"].rolling(7, min_periods=1).mean()
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    return df.dropna()

def train_predict_evaluate_xgboost(traindict, testdict, param_distributions, random_state):
    metrics = []
    for cat in traindict.keys():
        train = feature_engineering(traindict[cat])
        test = feature_engineering(testdict[cat])
        if len(train) < 10 or len(test) < 2:
            continue
        X_train = train.drop(columns=["units_sold", "date", "category"])
        y_train = train["units_sold"]
        X_test = test.drop(columns=["units_sold", "date", "category"])
        y_test = test["units_sold"]
        tscv = TimeSeriesSplit(n_splits=3)
        xgb = XGBRegressor(objective="reg:squarederror", random_state=random_state)
        search = RandomizedSearchCV(
            xgb, param_distributions, n_iter=10, cv=tscv,
            scoring="neg_mean_squared_error", random_state=random_state, n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        preds = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1e-8))) * 100
        baseline_pred = np.roll(y_test.values, 1)
        baseline_mae = mean_absolute_error(y_test.values[1:], baseline_pred[1:])
        metrics.append({
            "category": cat,
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "baseline_mae": baseline_mae
        })
    return pd.DataFrame(metrics)
