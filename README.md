# 📊 Inteligência Artificial na Gestão de Carteiras de FIIs e REITs

Este repositório contém o código desenvolvido no âmbito do **Programa de Iniciação Científica (PIBICVOL/UFS)** para a pesquisa **"Aplicação de Machine Learning na montagem de carteiras de FIIs e REITs: Maximizando a Eficiência no Reinvestimento de Proventos"**.

O projeto tem como objetivo aplicar **técnicas de aprendizado de máquina** para auxiliar na **automação do reinvestimento de proventos** em Fundos Imobiliários (FIIs) no Brasil e Real Estate Investment Trusts (REITs) nos EUA, visando estratégias de alocação mais eficientes.

---

## 📌 Objetivos

- Construir um **conjunto de dados robusto** a partir de fontes públicas (Yahoo Finance).
- Aplicar **modelos de regressão e classificação** para análise de performance de ativos.
- Avaliar a **viabilidade de estratégias automatizadas** de reinvestimento.
- Investigar o impacto de técnicas supervisionadas e não supervisionadas na formação de carteiras.

---

## 🛠️ Metodologia

1. **Coleta e Preparação de Dados**
   - Extração de dados via `yahooquery`.
   - Conjuntos de dados em granularidade mensal, semanal e diária.
   - Criação de features adicionais (média móvel, volatilidade etc).
   - Seleção de variáveis com **LLM assistido** e **SelectKBest (scikit-learn)**.

2. **Treinamento dos Modelos**
   - **Regressão**: avalia previsibilidade do retorno dos ativos.
   - **Classificação Não Supervisionada**: K-means e Agglomerative Clustering.
   - **Classificação Supervisionada**: métricas de mercado (crescimento e Dividend Yield) como rótulos.

3. **Avaliação**
   - Regressão: R² = 0.84  
   - K-means: Silhouette Score = 0.96  
   - Classificação supervisionada: acurácia até 81%, corrigida para 72.2% após balanceamento de classes.

---

## 🚀 Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/extract_data_B3.git
   cd extract_data_B3-main


Crie e ative um ambiente virtual:

- python -m venv venv
- source venv/bin/activate   # Linux/Mac
- venv\Scripts\activate      # Windows


Instale as dependências:

- pip install -r requirements.txt


Execute os notebooks da pasta notebooks/ ou os scripts em src/.

📈 Resultados

- Regressão apresentou forte correlação entre variáveis históricas e retornos.

- Clustering produziu clusters bem definidos, mas com pouca correlação com desempenho real.

- Classificação supervisionada apresentou bons resultados após balanceamento das classes.

🔮 Trabalhos Futuros

- Integrar modelos de regressão em estratégias práticas de alocação.

- Criar uma interface interativa para uso por investidores.

- Expandir a base de dados e aplicar técnicas mais avançadas de Machine Learning.

📖 Referência

Este repositório é parte do relatório de Iniciação Científica:

"Inteligência Artificial na Gestão de Carteiras de FIIs e REITs: Maximizando a Eficiência no Reinvestimento de Proventos"
Universidade Federal de Sergipe (UFS) – PIBICVOL
