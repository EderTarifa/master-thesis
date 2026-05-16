# Script de limpieza post-descarga
import pandas as pd
import numpy as np

df = pd.read_parquet('data/FX_MIX.parquet')
log_ret = np.log(df).diff()

# Detectar outliers extremos (>30% log-return = ~35% price change)
mask = (log_ret.abs() > 0.30)
n_outliers = mask.sum().sum()
print(f'Outliers detectados (>30%): {n_outliers}')

# Reemplazar el precio del día anómalo por el precio del día anterior
# (esto convierte el retorno extremo en 0)
for ticker in df.columns:
    bad_dates = log_ret.index[mask[ticker]]
    for date in bad_dates:
        prev_date = df.index[df.index.get_loc(date) - 1]
        df.loc[date, ticker] = df.loc[prev_date, ticker]
        print(f'  Limpiado {ticker} en {date.date()}')

df.to_parquet('data/FX_MIX.parquet')
print('FX_MIX limpio guardado')