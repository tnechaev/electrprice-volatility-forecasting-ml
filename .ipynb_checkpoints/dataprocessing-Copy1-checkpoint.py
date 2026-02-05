import numpy as np
import pandas as pd

def build_features(X, Y=None, add_target=False, eps=1e-8):

    if add_target:
        Xs = X.sort_values('ID').reset_index(drop=True)
        Ys = Y.sort_values('ID').reset_index(drop=True)

        if not (Xs['ID'].values == Ys['ID'].values).all():
            raise ValueError("X and Y IDs do not match")

        df = Xs.copy()
        df['volatility'] = Ys['TARGET'].values
    else:
        df = X.copy()

    # ---- DAY_ID handling (unchanged) ----
    if not np.issubdtype(df['DAY_ID'].dtype, np.number):
        try:
            df['DAY_ID'] = pd.to_datetime(df['DAY_ID'])
        except Exception:
            pass

    # Sort for time ops
    df = df.sort_values(['COUNTRY','DAY_ID']).reset_index(drop=True)

    # ---- VOLATILITY ----
    if add_target:
        for lag in [1,3,7]:
            df[f'vol_lag{lag}'] = df.groupby('COUNTRY')['volatility'].shift(lag)

        for w in [7,30]:
            df[f'vol_roll_std_{w}'] = df.groupby('COUNTRY')['volatility'] \
                .transform(lambda x: x.rolling(w, min_periods=3).std().shift(1))

            df[f'vol_roll_mean_{w}'] = df.groupby('COUNTRY')['volatility'] \
                .transform(lambda x: x.rolling(w, min_periods=3).mean().shift(1))

    # ---- FLAGS / SPREADS ----
    df['IS_FR'] = (df['COUNTRY'] == 'FR').astype(int)

    df['LOAD_IMBALANCE'] = df['DE_RESIDUAL_LOAD'] - df['FR_RESIDUAL_LOAD']
    df['WIND_IMBALANCE'] = df['DE_WINDPOW'] - df['FR_WINDPOW']
    df['SOLAR_IMBALANCE'] = df['DE_SOLAR'] - df['FR_SOLAR']
    df['NUCLEAR_IMBALANCE'] = df['FR_NUCLEAR'] - df['DE_NUCLEAR']

    df['FLOW_PRESSURE'] = df['DE_FR_EXCHANGE'] - df['FR_DE_EXCHANGE']
    df['TOTAL_FLOW'] = df['DE_FR_EXCHANGE'].abs() + df['FR_DE_EXCHANGE'].abs()

    df['DE_RESIDUAL_STRESS'] = df['DE_RESIDUAL_LOAD'] / (df['DE_CONSUMPTION'] + eps)
    df['FR_RESIDUAL_STRESS'] = df['FR_RESIDUAL_LOAD'] / (df['FR_CONSUMPTION'] + eps)

    df['GAS_COAL_SPREAD'] = df['GAS_RET'] - df['COAL_RET']
    df['CARBON_PRESSURE'] = df['CARBON_RET'] * (df['DE_COAL'] + df['DE_LIGNITE'])

    # ---- REGIMES ----
    df['GAS_RET_30m'] = df.groupby('COUNTRY')['GAS_RET'] \
        .transform(lambda x: x.rolling(30, min_periods=1).mean().shift(1))

    df['LOAD_TREND_30'] = df.groupby("COUNTRY")['DE_RESIDUAL_LOAD'] \
        .transform(lambda x: x.rolling(30, min_periods=10).mean().shift(1))

    df['REL_RENEWABLE'] = (df['DE_WINDPOW'] + df['DE_SOLAR']) - \
                          (df['FR_WINDPOW'] + df['FR_SOLAR'])

    # ---- WEATHER ----
    for c in ['DE_TEMP','FR_TEMP','DE_WIND','FR_WIND','DE_RAIN','FR_RAIN']:
        m = df.groupby('COUNTRY')[c].transform(lambda x: x.rolling(7, min_periods=3).mean())
        df[f'{c}_ANOM'] = df[c] - m

    # ---- EXTRA ROLLING ----
    for col in ['DE_RESIDUAL_LOAD','FR_RESIDUAL_LOAD','DE_WINDPOW',
                'FR_WINDPOW','DE_CONSUMPTION','FR_CONSUMPTION']:
        for w in [3,7,30]:
            df[f'{col}_rm_{w}'] = df.groupby('COUNTRY')[col] \
                .transform(lambda x: x.rolling(w, min_periods=2).mean().shift(1))
            df[f'{col}_std_{w}'] = df.groupby('COUNTRY')[col] \
                .transform(lambda x: x.rolling(w, min_periods=2).std().shift(1))

    df['LOADxGAS'] = df['LOAD_IMBALANCE'] * df['GAS_RET_30m']

    for c in ['DE_RESIDUAL_LOAD','FR_RESIDUAL_LOAD','LOAD_IMBALANCE','TOTAL_FLOW']:
        df[f'{c}_day_rank'] = df.groupby('DAY_ID')[c] \
            .transform(lambda x: x.rank(method='average')/len(x))

    return df