from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import requests
import json

def check_latest_model_date():
    """
    Check MLflow for the latest model training date
    Return True if it's been 3+ months, False otherwise
    """
    try:
        # MLflow API call to get latest experiment runs
        mlflow_url = "http://mlflow:5000/api/2.0/mlflow/runs/search"
        response = requests.post(
            mlflow_url,
            json={
                "experiment_ids": ["0"],  # Adjust with your experiment ID
                "order_by": ["start_time DESC"],
                "max_results": 1
            }
        )
        
        if response.status_code == 200:
            runs = response.json().get('runs', [])
            if runs:
                latest_run = runs[0]
                latest_timestamp = latest_run['info']['start_time'] / 1000  # Convert to seconds
                latest_date = datetime.fromtimestamp(latest_timestamp)
                
                three_months_ago = datetime.now() - timedelta(days=90)
                
                print(f"Latest model trained: {latest_date}")
                print(f"Three months ago: {three_months_ago}")
                
                if latest_date < three_months_ago:
                    print("Triggering training - latest model is older than 3 months")
                    return True
                else:
                    print("Skipping training - latest model is less than 3 months old")
                    return False
        
        # If no models found or API call fails, trigger training
        print("No previous models found or API error - triggering training")
        return True
        
    except Exception as e:
        print(f"Error checking MLflow: {e} - triggering training")
        return True

def get_training_date():
    """Get today's date for training in the required format"""
    return datetime.now().strftime("%Y-%m-%d")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="scheduled_training_dag",
    description="Training pipeline triggered when latest model is 3+ months old",
    default_args=default_args,
    start_date=datetime(2023, 1, 1),
    schedule_interval="@daily",  # Check daily
    catchup=False,
    tags=["training", "ml"],
) as dag:

    start = EmptyOperator(task_id="start")
    
    # Wait for data pipeline to complete first
    wait_for_data = ExternalTaskSensor(
        task_id="wait_for_data_pipeline",
        external_dag_id="data_pipeline_dag",  # Your data pipeline DAG
        external_task_id="gold_table",        # Wait for gold processing to complete
        mode="reschedule",
        timeout=3600,  # 1 hour timeout
        poke_interval=300,  # Check every 5 minutes
        retries=2,
    )
    
    # Use ShortCircuitOperator instead of PythonOperator
    check_trigger = ShortCircuitOperator(
        task_id="check_latest_model_age",
        python_callable=check_latest_model_date,
    )
    
    # Get current date for training
    get_date = PythonOperator(
        task_id="get_training_date",
        python_callable=get_training_date,
    )

    train_xgb = BashOperator(
        task_id="run_XGB",
        bash_command=(
            "cd /app && python /app/model_training_pipeline.py "
            "--train_date {{ ti.xcom_pull(task_ids='get_training_date') }}"
        ),
    )

    end = EmptyOperator(task_id="end")

    # Updated flow: Wait for data → Check trigger → Training
    start >> wait_for_data >> check_trigger >> get_date >> train_xgb >> end