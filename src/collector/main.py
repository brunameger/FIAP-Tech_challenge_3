import os
import requests
import pandas as pd
from google.cloud import bigquery
from flask import Flask, jsonify

app = Flask(__name__)

# Configurações via variáveis de ambiente
BQ_TABLE = os.environ.get("BQ_TABLE", "fiap-tech3.tc_dataset.ibov")
B3_URL = os.environ.get(
    "B3_URL",
    "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/eyJsYW5ndWFnZSI6InB0LWJyIiwicGFnZU51bWJlciI6MSwicGFnZVNpemUiOjIwLCJpbmRleCI6IklCT1YiLCJzZWdtZW50IjoiMSJ9"
)
USE_STREAMING = os.environ.get("USE_STREAMING", "true").lower() == "true"

client = bigquery.Client()


def fetch_ibov(url: str):
    """Busca dados do IBOV e retorna um DataFrame limpo"""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Cria DataFrame a partir do resultado
    df = pd.DataFrame(data.get("results", [])).copy()

    # Converte a data do header
    header_date = data.get("header", {}).get("date")
    if header_date:
        df["data_referencia"] = pd.to_datetime(header_date, dayfirst=True).date()

    # Renomeia colunas se existirem
    rename = {
        "code": "cod",
        "asset": "asset",
        "type": "type",
        "part": "part",
        "theoricalQty": "theoricalQty",
        "theoreticalQty": "theoricalQty",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Converte tipos numéricos
    if "part" in df.columns:
        df["part"] = pd.to_numeric(df["part"], errors="coerce")
    if "theoricalQty" in df.columns:
        df["theoricalQty"] = pd.to_numeric(df["theoricalQty"], errors="coerce")

    # Seleciona colunas finais
    cols = [c for c in ["cod", "asset", "type", "part", "theoricalQty", "data_referencia"] if c in df.columns]
    df = df[cols]

    return df


def load_to_bq_append(df: pd.DataFrame, table_id: str):
    """Insere os dados no BigQuery, tratando NaN e datas"""
    if df.empty:
        return {"rows": 0}

    # Copia o DataFrame e trata NaN e datas
    df_clean = df.copy()
    df_clean = df_clean.where(pd.notnull(df_clean), None)  # substitui NaN por None

    # Converte datas para string ISO
    if "data_referencia" in df_clean.columns:
        df_clean["data_referencia"] = df_clean["data_referencia"].astype(str)

    if USE_STREAMING:
        # Streaming inserts
        rows = df_clean.to_dict(orient="records")
        errors = client.insert_rows_json(table_id, rows)
        if errors:
            raise RuntimeError(f"BigQuery streaming errors: {errors}")
        return {"rows": len(rows)}
    else:
        # Batch load
        job = client.load_table_from_dataframe(
            df_clean,
            table_id,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        )
        job.result()
        dest = client.get_table(table_id)
        return {"rows": len(df_clean), "table_rows": dest.num_rows}


@app.route("/collect", methods=["POST"])
def collect_endpoint():
    """Endpoint do Cloud Run para coletar e enviar dados"""
    try:
        df = fetch_ibov(B3_URL)
        res = load_to_bq_append(df, BQ_TABLE)
        return jsonify({"status": "ok", "detail": res})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
