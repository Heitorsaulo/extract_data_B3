import pickle
import pandas as pd
import yahooquery as yq

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

def retrieve_ticker(lista_stocks):
    dataJson = list()

    for Stock in lista_stocks:
        try:
            dataJson.append(yq.Ticker(Stock))
        except Exception as e:
            print(e)
    return dataJson

def concat_data(all_stocks_data):
    df_stocks = pd.concat(all_stocks_data).drop_duplicates().reset_index(drop=True)
    df_stocks = df_stocks.fillna(0)
    return df_stocks

def fetch_stock_data(lista_reits):
    """
    Fetches financial data, summary profiles, and dividend information for REITs.
    """
    tickers = retrieve_ticker(lista_reits)

    with open("../../colunas_extraction/colunas_random_forest_classification.pkl", "rb") as f:
        feature_random_forest = pickle.load(f)

    all_stocks_data = list()
    for ticker in tickers:
        try:
            balance_sheet = ticker.get_financial_data(types=feature_random_forest, frequency='q', trailing=False)
            if isinstance(balance_sheet, str):
                continue

            start_date, end_date = balance_sheet['asOfDate'].iloc[[0, -1]]

            historic_value = ticker.history(start=start_date, end=end_date, interval='1wk')

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
                .assign(date=lambda df: pd.to_datetime(df['date']),  # Converte para datetime
                        month=lambda df: df['date'].dt.month,
                        year=lambda df: df['date'].dt.year)  # Extrai o ano
            )

            complete_values = pd.merge(balance_sheet, historic_value, on=['year', 'month'], how='inner')

            complete_values.drop(columns=['date_x'], inplace=True)  # Remove data duplicada
            complete_values.rename(columns={'date_y': 'date'}, inplace=True)

            complete_values['symbol'] = historic_value['symbol']

            complete_values['date'] = pd.to_datetime(complete_values['date'])
            complete_values.set_index('date', inplace=True)

            all_stocks_data.append(complete_values)

        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")

    return concat_data(all_stocks_data)

if __name__ == '__main__':
    screeners = ['reit_diversified',
        'reit_healthcare_facilities',
        'reit_hotel_motel',
        'reit_industrial',
        'reit_office',
       'reit_residential',
       'reit_retail']
    print("Starting Stocks names extraction...")
    reit_list = extract_reit_data(screeners)
    #FIIs = pd.read_csv('data/statusinvest-busca-avancada.csv', sep=';')
    lista_tickers = []
    for tickers in reit_list:
        lista_tickers.append(tickers)
    print("Finished Stocks names extraction...")
    print("---------------------------------")
    print("Starting Stocks data extraction...")
    reit_df = fetch_stock_data(lista_tickers)
    reit_df.to_csv('data/reit_data_random_forest_col.csv', index=False)
    print("FIIs data extracted and saved to 'reit_data.csv'")