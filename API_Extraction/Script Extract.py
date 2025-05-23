# %%
import json

import datetime as dt
import pandas as pd
import requests
#7QP5EEWK8EMX6MNU API KEY

# ## Codigo salvo
# url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&symbol=BTLG11.SA&apikey=7QP5EEWK8EMX6MNU'
# r = requests.get(url)
# data = r.json()
# 
# df = pd.json_normalize(data)
# print(data)
# df
# 
# API link para extrair noticias sobre o ativo requisitado

# # Diretorio dos dados extraidos
# 
# D:\ProjetoFII\dadosColetadosFii
# 

jsonFiisImport = pd.read_json(r'C:\Users\Heitor\PycharmProjects\DadosFii\Fii.json')
ERROR_MSG = 'Error Message'
# # Fundos que já foram feitos requisição: 0 a 65

dataJson = list()

for num in range(65,80):
    stringFii = jsonFiisImport['title'][num]
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={stringFii}.SA&apikey=7QP5EEWK8EMX6MNU'
    print(url)
    r = requests.get(url)
    data = r.json()
    dataJson.append(data)


dfDataFii = pd.DataFrame(dataJson)



# # Filtrar todos os fundos que o request na API conseguiu ter acesso aos dados:

# %%

count = 0
listaIsNa = pd.isna(dfDataFii[ERROR_MSG])
for data in dfDataFii[ERROR_MSG]:
    if not listaIsNa[count]:
        dfDataFii.drop(index=count)
    count += 1

# %%
dfDataFii.drop(columns=ERROR_MSG)


# %%
len(dfDataFii)

# %%
dfAuxDataFii = list()
dfAuxMetaDataFii = list()
for i in range(0, len(dfDataFii)):
    dfAuxDataFii.append(pd.DataFrame(dfDataFii['Monthly Adjusted Time Series'][dfDataFii.index[i]]))
    dfAuxMetaDataFii.append(dfDataFii['Meta Data'][dfDataFii.index[i]]['2. Symbol']) 

# %%
for i in range(0,len(dfAuxDataFii)):
    dfAuxDataFii[i] = dfAuxDataFii[i].transpose()
    dfAuxDataFii[i].index.name = dfAuxMetaDataFii[i]
# %% [markdown]
# ## Salvando todos os dados em uma pasta

# %%
for i in range(0,len(dfAuxDataFii)):
    dfAuxDataFii[i].to_csv(f'D:\ProjetoFII\dadosColetadosFii\dadosFii{dfAuxMetaDataFii[i]}.csv')


