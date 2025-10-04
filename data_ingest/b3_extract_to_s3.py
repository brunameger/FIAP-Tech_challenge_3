#!/usr/bin/env python3
import os, sys, datetime, requests, logging, io
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import boto3
from boto3.s3.transfer import TransferConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUCKET = os.environ.get("S3_BUCKET", "tech-challenge3")
REGION = os.environ.get("AWS_REGION", "sa-east-1")
MULTIPART_THRESHOLD = 8 * 1024 * 1024  # 8 MB

def upload_file_to_s3(local_path, s3_key):
    s3 = boto3.client('s3', region_name=REGION)
    config = TransferConfig(multipart_threshold=MULTIPART_THRESHOLD, max_concurrency=4)
    extra_args = {}

    logger.info("Uploading %s to s3://%s/%s", local_path, BUCKET, s3_key)
    s3.upload_file(local_path, BUCKET, s3_key, ExtraArgs=extra_args if extra_args else None, Config=config)
    logger.info("Upload finished")

def main(date_override=None):
    hoje = date_override if date_override else datetime.date.today().strftime("%Y%m%d")
    tmp_dir = f"/tmp/raw/date={hoje}"
    os.makedirs(tmp_dir, exist_ok=True)

    csv_path = f"{tmp_dir}/b3.csv"
    parquet_path = f"{tmp_dir}/b3.parquet"
    url = f"https://www.b3.com.br/pesquisapregao/download?filelist={hoje}/{hoje}_TradeReport.csv"

    resp = requests.get(url, timeout=30)
    if resp.status_code != 200 or len(resp.content) == 0:
        logger.error("Erro ao baixar CSV (status %s). Ajuste data/manualmente.", resp.status_code)
        sys.exit(1)

    with open(csv_path, "wb") as f:
        f.write(resp.content)
    logger.info("CSV salvo localmente: %s", csv_path)

    # leitura
    df = pd.read_csv(csv_path, sep=";", encoding="latin1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    logger.info("Parquet gerado: %s", parquet_path)

    s3_key = f"raw/date={hoje}/b3.parquet"
    upload_file_to_s3(parquet_path, s3_key)

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)

