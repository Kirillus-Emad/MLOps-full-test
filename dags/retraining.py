"""Retraining DAG: manual trigger, S3 new-file check -> preprocess -> train/evaluate/track.

Trigger manually from Airflow UI (Play button) or CLI:
    airflow dags trigger retraining --conf '{"s3_bucket": "my-bucket", "s3_prefix": "incoming"}'

Pipeline:
    wait_for_new_file (S3KeySensor, wildcard)
        -> download_new_file (newest .csv, or conf s3_key)
        -> preprocess (src/preprocessing.py)
        -> train_evaluate_track (src/pipeline.py: train -> evaluate -> tracking -> register)

Setup:
    - AWS keys in UI: Admin -> Connections -> aws_default (no env keys needed)
    - Optional Airflow Variables: s3_data_bucket, s3_data_prefix
    - Worker needs: apache-airflow-providers-amazon, boto3, pyyaml + project deps
"""

from __future__ import annotations

import os
import sys

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AWS_CONN_ID = "aws_default"

DEFAULT_BUCKET = Variable.get(
    "s3_data_bucket",
    default_var="mlops-demo-bucket",
)

DEFAULT_PREFIX = Variable.get(
    "s3_data_prefix",
    default_var="incoming",
)


def _resolve_s3_target(context):
    """Resolve (bucket, prefix, explicit_key) from conf -> params -> defaults."""
    params = context.get("params", {}) or {}

    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None) or {}

    bucket = (
        conf.get("s3_bucket")
        or params.get("s3_bucket")
        or DEFAULT_BUCKET
    )

    prefix = (
        conf.get("s3_prefix")
        or params.get("s3_prefix")
        or DEFAULT_PREFIX
    )

    explicit_key = conf.get("s3_key") or params.get("s3_key")

    return bucket, prefix.strip("/"), explicit_key


def download_new_s3_file(**context):
    """Download newest CSV under prefix (or conf s3_key) to raw_path."""
    import yaml

    bucket, prefix, explicit_key = _resolve_s3_target(context)

    hook = S3Hook(aws_conn_id=AWS_CONN_ID)

    if explicit_key:
        key = explicit_key.lstrip("/")
    else:
        client = hook.get_conn()

        paginator = client.get_paginator("list_objects_v2")

        candidates = []

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                obj_key = obj["Key"]

                is_csv = obj_key.lower().endswith(".csv")
                is_empty = obj.get("Size", 0) == 0

                if is_csv and not is_empty:
                    candidates.append((obj["LastModified"], obj_key))

        if not candidates:
            raise FileNotFoundError(
                f"No .csv found in s3://{bucket}/{prefix}/"
            )

        candidates.sort(key=lambda item: item[0])
        key = candidates[-1][1]

    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Airflow isolated folder: airflow_data/data.csv (not local data/data.csv)
    local_path = PROJECT_ROOT / "airflow_data" / "data.csv"

    # Remove stale file/dir from previous failed runs
    # (prevents NotADirectoryError when data.csv exists as a folder)
    import shutil

    if local_path.is_dir():
        shutil.rmtree(local_path)

    if local_path.exists():
        local_path.unlink()

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading s3://{bucket}/{key} -> {local_path}")

    # Use get_object + write bytes instead of hook.download_file
    # (download_file treats local_path as dir when it exists as file)
    s3_client = hook.get_conn()

    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
    )

    with open(local_path, "wb") as file:
        file.write(response["Body"].read())

    task_instance = context["ti"]
    task_instance.xcom_push(key="s3_bucket", value=bucket)
    task_instance.xcom_push(key="s3_key", value=key)
    task_instance.xcom_push(key="local_path", value=str(local_path))

    return key


def run_preprocessing(**context):
    """Run preprocessing on airflow_data (isolated from local data/)."""
    os.chdir(PROJECT_ROOT)

    task_instance = context["ti"]

    s3_key = task_instance.xcom_pull(
        key="s3_key",
        task_ids="download_new_file",
    )

    local_path = task_instance.xcom_pull(
        key="local_path",
        task_ids="download_new_file",
    )

    print(f"Preprocessing downloaded file: {s3_key}")
    print(f"Local path: {local_path}")

    from src.preprocessing import preprocess

    preprocess(
        raw_path=str(PROJECT_ROOT / "airflow_data" / "data.csv"),
        processed_dir=str(PROJECT_ROOT / "airflow_data" / "processed"),
    )


def run_train_evaluate_track(**context):
    """Run pipeline on airflow_data: train -> evaluate -> tracking -> register."""
    os.chdir(PROJECT_ROOT)

    from src.pipeline import main

    main(
        processed_dir_override=str(PROJECT_ROOT / "airflow_data" / "processed"),
    )


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="retraining",
    description="Manual retraining when new CSV lands in S3",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    params={
        "s3_bucket": DEFAULT_BUCKET,
        "s3_prefix": DEFAULT_PREFIX,
        "s3_key": None,
    },
    tags=["mlops", "retraining"],
) as dag:

    wait_for_new_file = S3KeySensor(
        task_id="wait_for_new_file",
        bucket_name="{{ params.s3_bucket }}",
        bucket_key="{{ params.s3_prefix }}/*.csv",
        wildcard_match=True,
        aws_conn_id=AWS_CONN_ID,
        poke_interval=60,
        timeout=60 * 60 * 2,
        mode="reschedule",
        soft_fail=False,
    )

    download_new_file = PythonOperator(
        task_id="download_new_file",
        python_callable=download_new_s3_file,
    )

    preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=run_preprocessing,
    )

    train_evaluate_track = PythonOperator(
        task_id="train_evaluate_track",
        python_callable=run_train_evaluate_track,
    )

    wait_for_new_file >> download_new_file >> preprocess >> train_evaluate_track
