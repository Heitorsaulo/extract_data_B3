import pandas as pd
import yahooquery as yq
import numpy as np
from sklearn.impute import SimpleImputer

def extract_reit_data(screeners):
    """
    Extracts REIT data from Yahoo Finance screeners.
    """
    s = yq.Screener()
    lista_cod_reits = []
    reits = s.get_screeners(screeners, 100)
    pd_reits = pd.DataFrame(reits)
    lista_reits = []
    for reit_sector in screeners:
        for reits_data in pd_reits.iloc[9][reit_sector]:
            lista_reits.append(reits_data['symbol'])

    return lista_reits

def create_classification_target(data, growth_threshold=5):
    """Creates a classification target based on percentage growth."""
    data['classification'] = (data['percent_growth'] > growth_threshold).astype(int)
    return data

if __name__ == '__main__':
    screeners = ['reit_diversified',
                 'reit_healthcare_facilities',
                 'reit_hotel_motel',
                 'reit_industrial',
                 'reit_office',
                 'reit_residential',
                 'reit_retail']
    print("Starting REITs names extraction...")
    reit_list = extract_reit_data(screeners)
    print("Finished REIT names extraction...")

    data = pd.read_csv('../data/preprocessed_reit_data.csv')
    mean_col = data.drop(columns=['symbol']).mean()
    data.fillna(mean_col, inplace=True)

    for reits in reit_list:
        moment_stock = data[data['symbol'] == reits]
        moment_stock_index = 0
        for index, row in moment_stock.iterrows():
            try:
                diff = moment_stock['close_mean'].iloc[moment_stock_index + 1] - moment_stock['close_mean'].iloc[moment_stock_index]
                sum_div = moment_stock['dividends_mean'].iloc[moment_stock_index + 1] + moment_stock['dividends_mean'].iloc[moment_stock_index]
                money_per_quota = diff + sum_div
                percent_growth = (money_per_quota / moment_stock['close_mean'].iloc[moment_stock_index]) * 100
                data.loc[index, 'money_per_quota'] = money_per_quota
                data.loc[index, 'percent_growth'] = percent_growth
                moment_stock_index += 1
            except Exception as e:
                continue

    data = create_classification_target(data)

    data.to_csv('data/classification_dataset_att_random_forest_col.csv', index=False)
    print("Classification dataset saved to 'classification_dataset_att_random_forest_col.csv'")