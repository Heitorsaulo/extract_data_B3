import pandas as pd
from yahooquery import Ticker

# 1. Carrega o CSV que tem colunas 'ticker', 'year', 'month'
#ativos_comprar = pd.read_csv("data/ativos_comprar_resultado.csv")
ativos_vender = pd.read_csv("data/ativos_vender_resultado.csv")

# 2. Constrói 'start' e 'end' como strings 'YYYY-MM-DD'
ativos_vender['start_date'] = pd.to_datetime(ativos_vender[['year', 'month']].assign(day=1))
# 'end_date' será primeiro dia do mês seguinte:
ativos_vender['end_date'] = ativos_vender['start_date'] + pd.offsets.MonthBegin(1)

# 3. Agrupa por período para buscar preços
resultados = []

tickers = ativos_vender['symbol'].unique().tolist()
tkr = Ticker(tickers,
    asynchronous=True,
    retry=5,
    status_forcelist=[429, 500, 502, 503, 504],
    timeout=10,
    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0",
    max_workers=4
    )

# Busca os históricos
hist = tkr.history(start=ativos_vender['start_date'].min().strftime('%Y-%m-%d'),
                   end=ativos_vender['end_date'].max().strftime('%Y-%m-%d'))

results = []

for _, row in ativos_vender.iterrows():
    tk = row['symbol']
    start_ts = pd.to_datetime(row['start_date'])
    end_ts = pd.to_datetime(row['end_date'])
    if isinstance(hist, pd.DataFrame) and tk in hist.index.get_level_values(0):
        periodo = hist.loc[tk]
        perido_datetime = pd.to_datetime(periodo.index)
        try:
            # Aqui o índice e os limites são Timestamps compatíveis
            preco_inicio = periodo.loc[perido_datetime >= start_ts, 'close'].iloc[0]
            preco_fim = periodo.loc[perido_datetime < end_ts, 'close'].iloc[-1]
            pct = (preco_fim - preco_inicio) / preco_inicio * 100
            print(f"Ticker: {tk}, Start: {start_ts.date()}, Preço Início: {preco_inicio}, Preço Fim: {preco_fim}, Variação Percentual: {pct:.2f}%")
            results.append({
                'ticker': tk,
                'start': start_ts.date(),
                'price_start': preco_inicio,
                'price_end': preco_fim,
                'pct_variation': pct
            })
        except Exception as ex:
            print(f"Erro ao processar {tk}: {ex}")
    else:
        print(f"Dados não encontrados para {tk}")

ativos_vender_res = pd.DataFrame(results)
ativos_vender_res.to_csv("data/ativos_vender_resultados_valores.csv")
print(ativos_vender_res)
