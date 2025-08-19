#%%
import pandas as pd
import numpy as np
import yahooquery as yq
import yfinance as yf
#%%
pd_fiis = pd.read_csv('data/statusinvest-busca-avancada.csv', sep=';')
len(pd_fiis['TICKER'])
#%%
s = yq.Screener()

lista_screeners = ['reit_diversified',
    'reit_healthcare_facilities',
    'reit_hotel_motel',
    'reit_industrial',
    'reit_office',
   'reit_residential',
   'reit_retail']

lista_cod_reits = []
reits = s.get_screeners(lista_screeners, 100)
pd_reits = pd.DataFrame(reits)
json_reit_diversified = pd_reits.iloc[9]

for reit in json_reit_diversified:
    for item in reit:
        lista_cod_reits.append(item['symbol'])

len(lista_cod_reits)
#%% md
# Achar o simbolo de cada REIT
#%%
pd_reits.iloc[9]['reit_diversified'][1]
#%%
for reits_data in pd_reits.iloc[9]['reit_diversified']:
    print(reits_data)
    for index in range(len(reits_data)):
        print(index)
        print(reits_data[index])

#%%
hasgx = yq.Ticker('hasgx')
#%%
hasgx.fund_profile
#%%
hasgx_performance = hasgx.fund_performance
hasgx_performance
#%% md
### Buscar todos os Reits disponiveis na API
#%%
lista_reits = []
for reit_sector in lista_screeners:
    for reits_data in pd_reits.iloc[9][reit_sector]:
            lista_reits.append(reits_data['symbol'])

print(lista_reits)
#%% md
# Coleta os dados
tickers = yq.Ticker(lista_reits)

# Informações financeiras
financial_data = tickers.financial_data

# Informações de perfil (nome, setor, indústria)
summary_profile = tickers.summary_profile

# Informações de dividendos
dividends = tickers.key_stats

data = []

for ticker in lista_reits:
    try:
        f = financial_data.get(ticker, {})
        s = summary_profile.get(ticker, {})
        d = dividends.get(ticker, {})

        row = {
            'Ticker': ticker,
            'CompanyName': s.get('longName'),
            'Sector': s.get('sector'),
            'Industry': s.get('industry'),
            'PriceToEarnings': f.get('trailingPE'),
            'PriceToBook': f.get('priceToBook'),
            'DividendYield': f.get('dividendYield'),
            'MarketCap': f.get('marketCap'),
            'OperatingMargin': f.get('operatingMargins'),
            'DebtToEquity': f.get('debtToEquity'),
            'ReturnOnEquity': f.get('returnOnEquity'),
            'DividendRate': d.get('dividendRate'),
            'ForwardPE': d.get('forwardPE'),
            'Beta': d.get('beta'),
        }
        data.append(row)
        print("Inserido com sucesso: {}".format(row))
    except Exception as e:
        print(f"Erro ao processar {ticker}: {e}")

df = pd.DataFrame(data)
df.head()

#%%
import pickle

with open("../colunas_extraction/colunas_sugeridas_gpt.pkl", "rb") as f:
    feature_GPT = pickle.load(f)
#%%
import pickle

with open("../colunas_extraction/colunas_random_forest_classification.pkl", "rb") as f:
    feature_GPT = pickle.load(f)
#%%
dataJson = list()

for Reits in lista_reits:
    try:
        dataJson.append(yq.Ticker(Reits))
    except Exception as e:
        print(e)
len(dataJson)
#%%
def build_dataset_stocks(all_stock_data):
    for data_stocks in dataJson:
        try:
            balance_sheet = data_stocks.get_financial_data(types=feature_GPT, frequency='q', trailing=False)
            if isinstance(balance_sheet, str):
                continue

            start_date, end_date = balance_sheet['asOfDate'].iloc[[0, -1]]

            historic_value = data_stocks.history(start=start_date, end=end_date, interval='1d')

            balance_sheet = (
                balance_sheet
                .fillna(0)  # Substitui NaNs por 0
                .rename(columns={'asOfDate': 'date'})  # Renomeia a coluna
                .assign(date=lambda df: pd.to_datetime(df['date']),  # Converte para datetime
                        month=lambda df: df['date'].dt.month,
                        year=lambda df: df['date'].dt.year)  # Extrai o ano
            )

            historic_value = (
                historic_value
                .reset_index()  # Transforma índices em colunas para evitar problemas
                .assign(date=lambda df: pd.to_datetime(df['date']),# Converte para datetime
                        month=lambda df: df['date'].dt.month,
                        year=lambda df: df['date'].dt.year)  # Extrai o ano
            )

            complete_values = pd.merge(balance_sheet, historic_value, on=['year', 'month'], how='inner')

            complete_values.drop(columns=['date_x'], inplace=True)  # Remove data duplicada
            complete_values.rename(columns={'date_y': 'date'}, inplace=True)

            complete_values['symbol'] = historic_value['symbol']

            complete_values['date'] = pd.to_datetime(complete_values['date'])
            complete_values.set_index('date', inplace=True)

            all_stock_data.append(complete_values)
        except Exception as e:
            print(f"Error processing data for {data_stocks}: {e}")
#%%
all_info_stocks = []
build_dataset_stocks(all_info_stocks)
#%%
all_info_stocks[9]
#%%
df_stocks = pd.concat(all_info_stocks).drop_duplicates().reset_index(drop=True)
df_stocks = df_stocks.fillna(0)
df_stocks
#%%
df_stocks.to_csv('reits_3M_mergeForWeek.csv', index=False)
#%%
df_stocks = pd.read_csv('data/reits_3M_mergeForWeek.csv')
df_stocks.head()
#%%
def remover_colunas_correlacionadas(df):
    matrix_corr = df.corr().abs()
    remove_columns = set()

    for i_corr in range(len(matrix_corr.columns)):
        for j_corr in range(i_corr + 1, len(matrix_corr.columns)):
            if matrix_corr.iloc[i_corr, j_corr] > 0.87 and matrix_corr.columns[j_corr] != 'close':
                remove_columns.add(matrix_corr.columns[j_corr])

    return df.drop(columns = remove_columns)

def filtra_colunas_zero(df):
    mean = df.describe().loc['mean']
    cols_to_keep = mean[mean != 0].index
    return df[cols_to_keep]

def remover_colunas_nao_numericas(df):
    #colunas_nao_numericas = df_stocks.select_dtypes(exclude=['number']).columns.tolist()
    colunas_nao_numericas = ['periodType', 'currencyCode']
    return df.drop(colunas_nao_numericas, axis=1)
#%%
symbol = df_stocks['symbol']
df_fil = (df_stocks.pipe(remover_colunas_nao_numericas)
               .pipe(filtra_colunas_zero)
               .pipe(remover_colunas_correlacionadas)
               )
df_fil
#%%
df_fil['symbol'] = symbol
#%%
def merge_indices_stocks(all_stocks_param, df_indices_cleaned_param):
    df_indices_cleaned_param['year'] = pd.to_datetime(df_indices_cleaned_param['date']).dt.year
    df_indices_cleaned_param['month'] = pd.to_datetime(df_indices_cleaned_param['date']).dt.month
    common_dates = set(all_stocks_param['year']).intersection(set(df_indices_cleaned_param['year']))
    if not common_dates:
        print("Não há datas comuns entre os DataFrames.")
    else:
        merged_df = pd.merge(all_stocks_param, df_indices_cleaned_param, on=['year', 'month'], how='inner')

        return merged_df
#%%
merged_info = merge_indices_stocks(df_fil, pd.read_csv('data/monthly_indices.csv'))
#%%
merged_info
#%%
def calculate_monthly_stats(merged_info, symbol):
    merged_info_symbol = merged_info[merged_info['symbol'] == symbol].copy()

    numeric_columns = merged_info_symbol.select_dtypes(include=['number']).columns
    print(numeric_columns)
    merged_info_monthly = merged_info_symbol.groupby(['year', 'month'])[['open', 'close', 'dividends', 'volume']].agg(['mean', 'var'])

    merged_info_monthly.columns = ['_'.join(col).strip() for col in merged_info_monthly.columns]

    merged_info_monthly = merged_info_monthly.reset_index()

    return merged_info_monthly

symbols = merged_info['symbol'].unique()

monthly_stats = {}

for symbol in symbols:
    monthly_stats[symbol] = calculate_monthly_stats(df_fil, symbol)
    print(f"Calculated monthly stats for symbol: {symbol}")

print(monthly_stats[symbols[0]].head())
#%%
merged_info_monthly = pd.concat(monthly_stats.values()).reset_index(drop=True)
#%%
merge_indices_stocks(merged_info_monthly, )
#%%
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import Lasso

from sklearn.feature_selection import SelectFromModel
dados1 = np.array(imputer.fit_transform(merged_info_monthly.drop(columns=['year','month', 'open_mean'])))
dados2 = np.array(merged_info_monthly['close_mean'])

X_train, X_test, y_train, y_test = train_test_split(dados1, dados2, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train, y_train)
model = SelectFromModel(lasso, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)

model_regression = RandomForestRegressor(n_estimators=100, random_state=42)
cross_val_scores = cross_val_score(model_regression, X_train, y_train, cv=5)
print(f"Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cross_val_scores):.4f}")
print(f"conjunto de treinamento: {X_train}")

model_regression.fit(X_train, y_train)

y_pred = model_regression.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

resultados = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

print(resultados.head())
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb

imputer = SimpleImputer(strategy='mean')
merged_info_monthly_imputed = imputer.fit_transform(merged_info_monthly.drop(columns=['year','month', 'year_mean','month_mean', 'year_var','month_var']))

X = merged_info_monthly_imputed
y = merged_info_monthly['close_mean']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgboost = xgb.XGBRegressor(n_estimators=100,  # Number of boosting rounds
                            learning_rate=0.1, # Step size shrinkage
                            max_depth=3,       # Maximum depth of a tree
                            random_state=42)

xgboost.fit(X_train, y_train)

y_pred = xgboost.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
#%%
dados = np.array(df_fil.drop(columns=['symbol']))
num_clusters = 2
print(len(dados))
#%% md
### Silhouette Score 0.6963 || 3 clusters
### Silhouette Score 0.8415 || 2 clusters
#%%
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(dados)

clusters = kmeans.predict(df_fil.drop(columns=['symbol']))

silhouette_avg = silhouette_score(df_fil.drop(columns=['symbol']), clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")

resultados = pd.DataFrame(df_fil, columns=df_fil.columns)
resultados['Cluster'] = clusters

#%%
print(len(resultados['Cluster']))
#%%
resultados['Cluster'].value_counts().sort_index().plot(kind='bar', title='Distribuição dos Clusters')
#%% md
### Silhouette Score 0.2469 || 3 clusters
### Silhouette Score 0.1512 || 2 clusters
#%%
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=num_clusters, random_state=42)
gmm.fit(dados)

clusters = gmm.predict(df_fil.drop(columns=['symbol']))

silhouette_avg = silhouette_score(df_fil.drop(columns=['symbol']), clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")

resultados = pd.DataFrame(df_fil, columns=df_fil.columns)
resultados['Cluster'] = clusters

#%% md
### Silhouette Score 0.6275 || 3 clusters
### Silhouette Score 0.8548 || 2 clusters
#%%
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

agg = AgglomerativeClustering(n_clusters=num_clusters)
clusters = agg.fit_predict(dados)

silhouette_avg = silhouette_score(df_fil.drop(columns=['symbol']), clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")

resultados = pd.DataFrame(df_fil, columns=df_fil.columns)
resultados['Cluster'] = clusters

#%%
resultados['Cluster'].value_counts().sort_index().plot(kind='bar', title='Distribuição dos Clusters')
#%% md
## Visualização dos Clusters
### Possivelmente o cluster de numero 2 seria ignorar, já que a maioria dos ativos seriam "ignoraveis" ? --> pergunta
#%%
ativos_cluster0 = resultados[resultados['Cluster'] == 0]
ativos_cluster1 = resultados[resultados['Cluster'] == 1]
#%%
print(ativos_cluster0['symbol'].unique())
print(ativos_cluster1['symbol'].unique())
#%%
df_stocks
#%%
df_stocks[df_stocks['symbol'] == 'O'].describe()
#%%
ativos_cluster1