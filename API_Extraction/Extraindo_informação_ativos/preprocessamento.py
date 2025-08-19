# preprocessing.py
import pandas as pd

def remover_colunas_correlacionadas(df, threshold=0.87):
    """Removes columns with high correlation."""
    matrix_corr = df.corr().abs()
    remove_columns = set()

    for i_corr in range(len(matrix_corr.columns)):
        for j_corr in range(i_corr + 1, len(matrix_corr.columns)):
            if matrix_corr.iloc[i_corr, j_corr] > threshold and matrix_corr.columns[j_corr] != 'close':
                remove_columns.add(matrix_corr.columns[j_corr])

    return df.drop(columns=remove_columns)

def filtra_colunas_zero(df):
    """Filters out columns with all zero values."""
    mean = df.describe().loc['mean']
    cols_to_keep = mean[mean != 0].index
    return df[cols_to_keep]

def remover_colunas_nao_numericas(df, cols_to_remove = ['periodType', 'currencyCode']):
    """Removes non-numeric columns."""
    colunas_nao_numericas = cols_to_remove
    return df.drop(colunas_nao_numericas, axis=1)

def calculate_monthly_stats(merged_info, symbol, columns_for_stats = ['open', 'close', 'dividends', 'volume']):
    """Calculates monthly statistics for specified columns."""
    merged_info_symbol = merged_info[merged_info['symbol'] == symbol].copy()

    numeric_columns = merged_info_symbol.select_dtypes(include=['number']).columns
    columns_to_aggregate = [col for col in columns_for_stats if col in numeric_columns]

    merged_info_monthly = merged_info_symbol.groupby(['year', 'month'])[columns_to_aggregate].agg(['mean', 'var', 'min', 'max'])

    merged_info_monthly.columns = ['_'.join(col).strip() for col in merged_info_monthly.columns]

    merged_info_monthly = merged_info_monthly.reset_index()

    merged_info_symbol = merged_info_symbol.drop(columns=columns_to_aggregate)
    merged_info_symbol = merged_info_symbol.groupby(['year', 'month']).first().reset_index()

    merged_info_monthly = pd.merge(merged_info_monthly, merged_info_symbol, on=['year', 'month'], how='left')

    return merged_info_monthly

def merge_indices_stocks(all_stocks_param, df_indices_cleaned_param):
    """Merges stock data with indices data based on year and month."""
    df_indices_cleaned_param['year'] = pd.to_datetime(df_indices_cleaned_param['date']).dt.year
    df_indices_cleaned_param['month'] = pd.to_datetime(df_indices_cleaned_param['date']).dt.month
    common_dates = set(all_stocks_param['year']).intersection(set(df_indices_cleaned_param['year']))
    if not common_dates:
        print("Não há datas comuns entre os DataFrames.")
        return None
    else:
        merged_df = pd.merge(all_stocks_param, df_indices_cleaned_param, on=['year', 'month'], how='inner')
        return merged_df

if __name__ == '__main__':
    # Load data
    df_stocks = pd.read_csv('../data/reit_data_random_forest_col.csv')

    # Apply preprocessing steps
    symbol = df_stocks['symbol']
    df_fil = (df_stocks.pipe(remover_colunas_nao_numericas)
                   .pipe(filtra_colunas_zero)
                   )
    df_fil['symbol'] = symbol

    # Example usage of merge_indices_stocks (assuming you have 'monthly_indices.csv')
    # df_indices = pd.read_csv('monthly_indices.csv')
    # merged_info = merge_indices_stocks(df_fil, df_indices)

    # Example usage of calculate_monthly_stats
    symbols = df_fil['symbol'].unique()
    monthly_stats = {}
    for symbol in symbols:
        monthly_stats[symbol] = calculate_monthly_stats(df_fil, symbol)

    merged_info_monthly = pd.concat(monthly_stats.values()).reset_index(drop=True)
    merged_info_monthly.to_csv('data/preprocessed_reit_data_random_forest_col.csv', index=False)
    print("Preprocessed REIT data saved to 'preprocessed_reit_data.csv'")