from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

def load_ibov_table(table_id: str, limit: int = None):
    q = f"SELECT * FROM `{table_id}` ORDER BY data_referencia ASC"
    if limit:
        q = q.replace("ASC", "ASC LIMIT {}".format(limit))
    df = client.query(q).to_dataframe()
    return df