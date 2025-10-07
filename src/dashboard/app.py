import streamlit as st
import pandas as pd
import requests
import altair as alt
from src.utils.bq_utils import load_ibov_table

st.set_page_config(page_title="Dashboard IBOV ML", layout="wide")

st.title("📈 Dashboard de Predição IBOV")
st.markdown("Este dashboard utiliza dados da B3 armazenados no BigQuery e predições do modelo XGBoost.")

# Gráfico de predição
def plot_predictions(df_sel, preds, ativo_selecionado):
    df_merge = df_sel.merge(
        preds[preds["cod"] == ativo_selecionado],
        on=["cod", "data_referencia", "asset"],
        how="left"
    )

    # Converte colunas numéricas e remove NaN
    df_merge["theoricalQty"] = pd.to_numeric(df_merge["theoricalQty"], errors="coerce")
    df_merge["prediction"] = pd.to_numeric(df_merge["prediction"], errors="coerce")
    df_plot = df_merge.dropna(subset=["theoricalQty", "prediction"])

    # Cria gráfico interativo
    chart = (
        alt.Chart(df_plot)
        .transform_fold(
            ["theoricalQty", "prediction"],
            as_=["Tipo", "Valor"]
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("data_referencia:T", title="Data de Referência"),
            y=alt.Y("Valor:Q", title="Quantidade Teórica"),
            color=alt.Color("Tipo:N", title="Série"),
            tooltip=[
                alt.Tooltip("data_referencia:T", title="Data"),
                alt.Tooltip("Valor:Q", title="Valor"),
                alt.Tooltip("Tipo:N", title="Tipo"),
            ],
        )
        .properties(width=800, height=400)
    )
    st.altair_chart(chart, use_container_width=True)


# Leitura dos dados históricos do BigQuery
try:
    st.sidebar.write("🔄 Carregando dados históricos do BigQuery...")
    df_hist = load_ibov_table("fiap-tech3.tc_dataset.ibov", limit=500)
    st.sidebar.success("Dados carregados com sucesso!")
except Exception as e:
    st.sidebar.error(f"Erro ao carregar dados: {e}")
    st.stop()

df_hist["data_referencia"] = pd.to_datetime(df_hist["data_referencia"])
df_hist = df_hist.sort_values(["cod", "data_referencia"])

ativos = sorted(df_hist["cod"].unique())
ativo_selecionado = st.sidebar.selectbox("Selecione um ativo:", ativos)

# Filtra o ativo escolhido
df_sel = df_hist[df_hist["cod"] == ativo_selecionado]

# Gráfico do histórico
st.subheader(f"📊 Histórico do ativo {ativo_selecionado}")
chart_hist = (
    alt.Chart(df_sel)
    .mark_line(point=True)
    .encode(
        x="data_referencia:T",
        y="theoricalQty:Q",
        tooltip=["data_referencia", "theoricalQty"],
    )
    .properties(width=800, height=400)
)
st.altair_chart(chart_hist, use_container_width=True)

# Predição do Modelo
st.subheader("🤖 Predição do Modelo")

api_url = st.text_input(
    "URL da API de Predição (Cloud Run ou local)",
    "http://127.0.0.1:8080/predict"
)

if st.button("Gerar Previsão"):
    try:
        with st.spinner("Chamando a API de predição..."):
            response = requests.post(api_url)
            if response.status_code == 200:
                preds = pd.DataFrame(response.json())
                preds["data_referencia"] = pd.to_datetime(preds["data_referencia"])

                # Filtra o ativo escolhido
                pred_ativo = preds[preds["cod"] == ativo_selecionado]

                if len(pred_ativo) == 0:
                    st.warning("Nenhuma previsão disponível para este ativo.")
                else:
                    next_pred = pred_ativo.iloc[0]
                    st.success(f"📅 Previsão para {next_pred['data_referencia'].date()}")
                    st.metric(
                        label=f"Quantidade Teórica Prevista ({next_pred['cod']})",
                        value=f"{next_pred['prediction']:.2f}"
                    )

                    # Mostra gráfico histórico + ponto previsto
                    df_plot = df_sel.copy()
                    df_plot = df_plot.sort_values("data_referencia")

                    ultimo_valor = df_plot["theoricalQty"].iloc[-1]
                    chart = (
                        alt.Chart(df_plot)
                        .mark_line(point=True)
                        .encode(
                            x="data_referencia:T",
                            y="theoricalQty:Q",
                            tooltip=["data_referencia", "theoricalQty"],
                        )
                        .properties(width=800, height=400)
                    )

                    point = pd.DataFrame({
                        "data_referencia": [next_pred["data_referencia"]],
                        "theoricalQty": [next_pred["prediction"]]
                    })

                    chart_pred = chart + alt.Chart(point).mark_point(
                        color="red", size=100
                    ).encode(
                        x="data_referencia:T",
                        y="theoricalQty:Q",
                        tooltip=["data_referencia", "theoricalQty"]
                    )

                    st.altair_chart(chart_pred, use_container_width=True)

            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"Erro ao gerar predição: {e}")

with st.expander("📋 Visualizar dados brutos do ativo selecionado"):
    st.dataframe(df_sel)
