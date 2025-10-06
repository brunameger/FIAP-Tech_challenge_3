# src/dashboard/app.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Previsão IBOV", layout="wide")
st.title("Dashboard de Previsão do IBOV")

# URL do endpoint Flask/Cloud Run
PREDICT_URL = st.text_input(
    "URL do endpoint de predição",
    value="http://127.0.0.1:8080/predict"
)

# Inicializa session_state para armazenar DataFrame
if "df" not in st.session_state:
    st.session_state.df = None

# Botão para carregar histórico e previsões
if st.button("Carregar histórico e prever"):
    with st.spinner("Buscando dados e gerando previsões..."):
        try:
            # Chama endpoint de predição
            response = requests.post(PREDICT_URL)
            response.raise_for_status()
            df_preds = pd.DataFrame(response.json())
            df_preds["data_referencia"] = pd.to_datetime(df_preds["data_referencia"])
            df_preds = df_preds.sort_values(["cod","data_referencia"])

            # Carrega histórico local (20 dias)
            df_hist = pd.read_csv("historico_ibov_20dias.csv")
            df_hist["data_referencia"] = pd.to_datetime(df_hist["data_referencia"])

            # Merge histórico com previsões
            df = pd.merge(
                df_hist,
                df_preds[["data_referencia","cod","prediction"]],
                on=["data_referencia","cod"],
                how="left"
            )

            st.session_state.df = df  # salva no session_state
            st.success("Dados carregados com sucesso!")

        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao conectar com o servidor de predição: {e}")
        except Exception as e:
            st.error(f"Erro ao processar os dados: {e}")

# Se df foi carregado, exibe gráficos
if st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("Tabela de Previsões x Histórico")
    st.dataframe(df)

    # Gráfico global interativo
    st.subheader("Gráfico de Previsões x Histórico")
    fig = px.line(
        df,
        x="data_referencia",
        y=["theoricalQty","prediction"],
        color="cod",
        labels={"value":"Quantidade Teórica","variable":"Tipo"},
        title="Previsões do IBOV vs Histórico"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dropdown ativo detalhado
    ativo_sel = st.selectbox("Selecionar ativo para detalhar gráfico:", df["cod"].unique())
    df_ativo = df[df["cod"] == ativo_sel]

    fig2 = px.line(
        df_ativo,
        x="data_referencia",
        y=["theoricalQty","prediction"],
        labels={"value":"Quantidade Teórica","variable":"Tipo"},
        title=f"Previsão detalhada: {ativo_sel}"
    )
    st.plotly_chart(fig2, use_container_width=True)
