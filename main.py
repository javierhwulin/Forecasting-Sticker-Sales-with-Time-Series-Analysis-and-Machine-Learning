import pandas as pd
import numpy as np
import requests
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import optuna
import warnings

warnings.filterwarnings('ignore')

###############################################################################
#                           Custom Transformers                               #
###############################################################################


class LagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lags=None, groupby_cols=None):
        if groupby_cols is None:
            groupby_cols = ['store', 'product']
        if lags is None:
            lags = [1, 7, 14, 28]
        self.lags = lags
        self.groupby_cols = groupby_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'num_sold' not in df.columns:
            df['num_sold'] = 0
        for lag in self.lags:
            df[f'sales_lag_{lag}'] = (
                df.groupby(self.groupby_cols)['num_sold'].shift(lag).fillna(0).values
            )
        return df


class RollingFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, windows=None, groupby_cols=None):
        if groupby_cols is None:
            groupby_cols = ['store', 'product']
        if windows is None:
            windows = [7, 14, 21, 28]
        self.windows = windows
        self.groupby_cols = groupby_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'num_sold' not in df.columns:
            # If 'num_sold' is missing (test set), use zeros for rolling features
            df['num_sold'] = 0
        for window in self.windows:
            df[f'sales_roll_mean_{window}'] = (
                df.groupby(self.groupby_cols)['num_sold']
                .transform(lambda x: x.rolling(window).mean().fillna(0).values)
            )
            df[f'sales_roll_std_{window}'] = (
                df.groupby(self.groupby_cols)['num_sold']
                .transform(lambda x: x.rolling(window).std().fillna(0).values)
            )
        return df


class SeasonalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'num_sold' not in df.columns:
            # If 'num_sold' is missing (test set), fill it with zeros
            df['num_sold'] = 0

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Initialize columns to handle cases where seasonal_decompose fails
        df['trend'] = 0
        df['seasonal'] = 0
        df['residual'] = 0

        try:
            result = seasonal_decompose(df['num_sold'], model='additive', period=7)
            df['trend'] = result.trend.fillna(0)
            df['seasonal'] = result.seasonal.fillna(0)
            df['residual'] = result.resid.fillna(0)
        except ValueError:
            pass

        df = df.reset_index()
        return df


class GDPFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gdp_map = None
        self.country_codes = {
            'Canada': 'CAN',
            'Finland': 'FIN',
            'Italy': 'ITA',
            'Kenya': 'KEN',
            'Norway': 'NOR',
            'Singapore': 'SGP',
        }

    def fetch_gdp_per_capita(self, country, year):
        alpha3 = self.country_codes.get(country)
        if not alpha3:
            return None
        url = f'https://api.worldbank.org/v2/country/{alpha3}/indicator/NY.GDP.PCAP.CD?date={year}&format=json'
        r = requests.get(url).json()
        try:
            return r[1][0]['value']
        except (IndexError, TypeError):
            return None

    def fetch_gdp_data(self, countries, years):
        gdp_data = {}
        for c in countries:
            for y in years:
                gdp_data[(c, y)] = self.fetch_gdp_per_capita(c, y)
        return gdp_data

    def fit(self, X, y=None):
        if 'date' in X.columns:
            X = X.copy()
            X['date'] = pd.to_datetime(X['date'], errors='coerce')
            years = X['date'].dt.year.dropna().unique()
        else:
            # If 'date' doesn't exist, can't fetch
            years = []
        countries = X['country'].dropna().unique()
        self.gdp_map = self.fetch_gdp_data(countries, years)
        return self

    def transform(self, X):
        df = X.copy()
        if 'date' not in df.columns:
            df['date'] = pd.NaT
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['gdp'] = df.apply(
            lambda r: self.gdp_map.get((r['country'], r['year']), np.nan),
            axis=1,
        )

        df['gdp'] = df['gdp'].fillna(
            df.groupby('country')['gdp'].transform('mean')
        )
        return df


# Custom transformer to wrap pandas one-hot encoding
class PandasOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.get_dummies(X, drop_first=True)


###############################################################################
#                           Helper Functions                                  #
###############################################################################


def load_data():
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')

    return train_df, test_df


def apply_irq_filter(df):
    q1 = df['num_sold'].quantile(0.25)
    q3 = df['num_sold'].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr
    return df[(df['num_sold'] > lb) & (df['num_sold'] < ub)]


def time_features(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    return df


###############################################################################
#                       Optuna Objective & Final Model                        #
###############################################################################


def objective(trial, X, y):
    # Hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

    numeric_features = ['year', 'month', 'dayofweek', 'is_weekend', 'gdp']
    cat_features = ['country', 'store', 'product']

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('target_enc', PandasOneHotEncoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', cat_pipeline, cat_features),
        ],
        remainder='drop',
    )

    # Preprocessing Pipeline (up to feature engineering)
    preprocessing_pipeline = Pipeline([
        ('lag_features', LagFeatures()),
        ('roll_features', RollingFeatures()),
        ('seasonal_features', SeasonalFeatures()),
        ('gdp_features', GDPFeatures()),
        ('preprocessor', preprocessor),
    ])

    # Apply preprocessing separately from model training
    X_preprocessed = preprocessing_pipeline.fit_transform(X, y)

    # Define the model
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
    )

    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    scores = cross_val_score(
        model, X_preprocessed, y, cv=tscv, scoring='neg_mean_absolute_error'
    )

    # Return the negative mean MAE as the objective metric
    return -np.mean(scores)


def train_final_model(X, y, best_params):
    numeric_features = ['year', 'month', 'dayofweek', 'is_weekend', 'gdp']
    cat_features = ['country', 'store', 'product']

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('target_enc', PandasOneHotEncoder()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', cat_pipeline, cat_features),
        ],
        remainder='drop',
    )

    pipeline = Pipeline(
        steps=[
            ('lag_features', LagFeatures()),
            ('roll_features', RollingFeatures()),
            ('seasonal_features', SeasonalFeatures()),
            ('gdp_features', GDPFeatures()),
            ('preprocessor', preprocessor),
            (
                'model',
                XGBRegressor(
                    n_estimators=best_params['n_estimators'],
                    max_depth=best_params['max_depth'],
                    learning_rate=best_params['learning_rate'],
                    subsample=best_params['subsample'],
                    colsample_bytree=best_params['colsample_bytree'],
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


def save_model(model, file_name='final_model.pkl'):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


###############################################################################
#                                   Main                                      #
###############################################################################


def main():
    train_df, test_df = load_data()

    # If the test set doesn't have 'num_sold', create it as NaN
    if 'num_sold' not in test_df.columns:
        test_df['num_sold'] = np.nan

    # Basic time features in both train & test
    train_df = time_features(train_df)
    test_df = time_features(test_df)

    # IRQ filtering on train
    train_df = apply_irq_filter(train_df)

    # Drop any remaining NaNs in train
    train_df = train_df.dropna(subset=['num_sold'])

    # Log-transform the target in train to handle skew
    train_df['num_sold'] = np.log1p(train_df['num_sold'])

    # Separate features/target for train
    X_train = train_df.drop(columns=['id'], errors='ignore')
    y_train = train_df['num_sold']

    # ---------------------
    # Optuna Hyperparam Tuning
    # ---------------------
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), n_trials=50
    )
    best_params = study.best_params
    print('Best Params:', best_params)

    # ---------------------
    # Train final pipeline
    # ---------------------
    final_pipeline = train_final_model(X_train, y_train, best_params)

    # Save the trained pipeline
    save_model(final_pipeline, 'final_model.pkl')
    print('Model saved as final_model.pkl')

    # ---------------------
    # Predict on test set
    # ---------------------
    # We'll drop 'id', but keep 'date', 'num_sold' so the pipeline won't crash
    X_test = test_df.drop(columns=['id'], errors='ignore')
    # Pipeline's transform will produce lags/rolling
    test_preds_log = final_pipeline.predict(X_test)

    # Undo log1p transform
    test_preds = np.expm1(test_preds_log)

    # Add predictions to test DataFrame
    test_df['predictions'] = test_preds

    # Submit these predictions
    test_df[['id', 'predictions']].to_csv('submission.csv', index=False)
    print('Predictions saved to submission.csv')


if __name__ == '__main__':
    main()
