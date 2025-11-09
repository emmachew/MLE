from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
from datetime import datetime, timedelta


# ------------------------
# HARDCODED VALUES - FIXED
# ------------------------
MODEL_FILE = "credit_model_2024_12_01"
CURRENT_SNAPSHOT = "2024-12-01"
BASELINE_SNAPSHOT = "2024-06-01"

INFERENCE_SCRIPT = "/app/utils/model_inference.py"
MONITORING_SCRIPT = "/app/utils/monitor_predictions.py"

# FIXED: Use the actual file naming pattern from your inference script
PRED_PATH_CUR = f"/app/datamart/gold/model_predictions/{MODEL_FILE}/{MODEL_FILE}_predictions_{CURRENT_SNAPSHOT.replace('-', '_')}.parquet"
PRED_PATH_BASE = f"/app/datamart/gold/model_predictions/credit_model_2024_06_01/credit_model_2024_06_01_predictions_{BASELINE_SNAPSHOT.replace('-', '_')}.parquet"
MONITOR_OUTPUT = f"/app/datamart/gold/model_monitoring/{MODEL_FILE}"
LABELS_PATH = "/app/datamart/gold/label_store"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
}


with DAG(
    dag_id="batch_inference_and_monitoring",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@monthly",
    catchup=False,
    default_args=default_args,
) as dag:

    start = EmptyOperator(task_id="start")

    # ------------------------
    # ✅ Inference task
    # ------------------------
    run_inference = BashOperator(
        task_id="run_inference",
        bash_command=f"""
    python {INFERENCE_SCRIPT} "{CURRENT_SNAPSHOT}" "{MODEL_FILE}.pkl"
        """,
        execution_timeout=timedelta(minutes=30),
    )

    # ------------------------
    # ✅ Monitoring task - FIXED
    # ------------------------
    run_monitor = BashOperator(
        task_id="run_prediction_monitor",
        bash_command=f"""
    python {MONITORING_SCRIPT} \
        --model_label {MODEL_FILE}.pkl \
        --snapshotdate {CURRENT_SNAPSHOT} \
        --baseline_date {BASELINE_SNAPSHOT} \
        --pred_path {PRED_PATH_CUR} \
        --baseline_path {PRED_PATH_BASE} \
        --output_dir {MONITOR_OUTPUT} \
        --labels_path {LABELS_PATH}
            """,
            execution_timeout=timedelta(minutes=30),
        )

    end = EmptyOperator(task_id="end")

    start >> run_inference >> run_monitor >> end