# Tech Challenge 3 – Previsão de Quantidade Teórica de Ativos IBOV

Este projeto é parte do Tech Challenge 3 da FIAP e consiste em **uma aplicação de Machine Learning** para prever a quantidade teórica de ativos do IBOVESPA. Ele inclui coleta de dados, armazenamento em BigQuery, treinamento de modelo XGBoost e um dashboard visual em Streamlit.

---

## 📌 Objetivos do Projeto

- Coletar dados do IBOV em **tempo real** via API da B3.
- Armazenar os dados coletados em **BigQuery**.
- Criar um modelo de **Machine Learning (XGBoost)** para prever a quantidade teórica do próximo dia.
- Disponibilizar o modelo via **API FastAPI**.
- Criar um **dashboard em Streamlit** para visualização de histórico e predições.

---

## 🏗 Estrutura do Projeto

```
.
├── Dockerfile
├── requirements.txt
├── README.md
├── src
│   ├── collector
│   │   └── main.py          # Coleta dados da B3 e envia para BigQuery
│   ├── predict
│   │   └── serve.py         # API FastAPI para predição
│   ├── trainer
│   │   └── train.py         # Script de treino do modelo XGBoost
│   ├── dashboard
│   │   └── app.py           # Dashboard Streamlit
│   └── utils
│       └── bq_utils.py      # Funções auxiliares para ler BigQuery
```

---

## ⚡ Tecnologias Utilizadas

- Python 3.11  
- FastAPI (API de predição)  
- Flask (coletor de dados)  
- Streamlit (dashboard)  
- XGBoost (modelo de ML)  
- Google Cloud:
  - BigQuery (armazenamento de dados)
  - Cloud Storage (armazenamento do modelo)
  - Cloud Run (deploy da API)
  - Cloud Scheduler (execução periódica da coleta)

---

## 📥 Instalação e Configuração

1. Clone o repositório:

```bash
git clone https://github.com/seu_usuario/tech-challenge3.git
cd tech-challenge3
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale dependências:

```bash
pip install -r requirements.txt
```

4. **Autenticação com Google Cloud**

Para acessar BigQuery, Cloud Storage e Cloud Run, você precisa de um **service account**:

- Baixe o **arquivo JSON da chave** do service account.
- Configure a variável de ambiente para autenticação:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/caminho/para/sua-chave.json"  # Linux/Mac
set GOOGLE_APPLICATION_CREDENTIALS="C:\caminho\para\sua-chave.json"  # Windows
```

Isso permitirá que os scripts Python acessem BigQuery e Cloud Storage de forma autenticada.


---

## 🤖 Treinamento do Modelo

O modelo XGBoost é treinado com `src/trainer/train.py`:

```bash
python -m src.trainer.train
```

- Utiliza **TimeSeriesSplit** para validação temporal.
- Cria features de lags (`theor_lag1`, `theor_lag2`), média móvel (`roll_mean_3`) e data (`dow`, `month`) e `cod_cat` (ativo categórico).
- Salva o modelo em **Cloud Storage** (`gs://fiap-tech3-models/ibov_xgb_v1.joblib`).

---

## 🚀 API de Predição

O arquivo `src/predict/serve.py` disponibiliza um endpoint FastAPI:

```
POST /predict
```

- Retorna a previsão do **próximo dia** para todos os ativos.  
- O modelo é carregado diretamente do **bucket do GCS**.  
- Cada ativo inclui:
  - `cod`: código do ativo
  - `asset`: nome do ativo
  - `data_referencia`: data prevista
  - `prediction`: quantidade teórica prevista

Executando localmente:

```bash
python -m src.predict.serve
```

---

## 📊 Dashboard Streamlit

O dashboard (`src/dashboard/app.py`) permite:

- Selecionar um ativo do IBOV.
- Visualizar histórico de quantidade teórica.
- Gerar a previsão do próximo dia usando a API FastAPI.
- Visualizar o ponto previsto no gráfico.

Executando:

```bash
streamlit run src/dashboard/app.py
```

---

## 🧩 Arquitetura Geral

```
IBOV API → Collector (Flask) → BigQuery → Trainer (XGBoost) → Cloud Storage → API (FastAPI) → Dashboard (Streamlit)
```

- **Collector**: coleta e envia dados.
- **BigQuery**: banco de dados histórico.
- **Trainer**: treina e envia modelo ao GCS.
- **API FastAPI**: fornece predições.
- **Dashboard**: interface visual para histórico e previsão.

---

## 📁 Links Úteis

- [BigQuery](https://console.cloud.google.com/bigquery)
- [Cloud Storage](https://console.cloud.google.com/storage)
- [Cloud Run](https://console.cloud.google.com/run)
- [Streamlit](https://streamlit.io/)

---

## 📝 Autor

Alexandre Ghirello Cabestré
Briseyda Carolina Chambi Vargas Cardona
Bruna de Souza Meger
Matheus Brum Pereira
Roseane de Souza Silva
