# Section 1: Directory Tree 📂
```
assign_2/
│
├── data/                       # Datasets
│   ├── feature_clickstream.csv
│   ├── features_attributes.csv
│   ├── features_financials.csv
│   └── lms_loan_daily.csv
│
├── datamart/                   # Medallion data warehouse
│   ├── bronze/ (csv, by year and month)
│   ├── silver/ (Parquet)
│   └── gold/   (Parquet, feature_store/, label_store/)
│
├── airflow/                    # Airflow setup + config
├── dags/                       # DAGs: data pipeline, training, monitoring
│
├── notebooks/                  # Dev notebooks
│
├── mlflow/                     # MLflow tracking/experiments
│
├── utils/			# PySpark + ML training scripts
│   ├── processing_bronze_table.py
│   ├── processing_silver_table.py
│   ├── processing_gold_table.py
│   ├── model_training_LR.py
│   ├── model_training_XG.py
│   └── model_training_RF.py
│
├── docker-compose.yaml         # Airflow + MLflow orchestration
│
├── data_processing_pipeline.py     # Bronze/Silver/Gold pipeline (ETL)
├── model_training_pipeline.py        # Training + MLflow registration
├── inference_and_monitoring_pipeline.py      # Batch/online inference + monitoring
│
└── README.md

```
# Section 2: How to Run 
## 1️⃣ Start Environment

Make sure you have Docker + Docker Compose installed.  
Build and start all services (Airflow, MLflow, JupyterLab):
```bash
docker-compose up --build
```
Once started:  
| Service                | URL                                            |
| ---------------------- | ---------------------------------------------- |
| **Airflow Web UI**     | [http://localhost:8080](http://localhost:8080) |
| **MLflow Tracking UI** | [http://localhost:5000](http://localhost:5000) |
| **JupyterLab**         | [http://localhost:8888](http://localhost:8888) |


## 2️⃣ Run Data Pipeline

### Option A – via Airflow (Recommended)
Airflow DAGs are located in /dags:  
| DAG                                 | Purpose                                   |
| ----------------------------------- | ----------------------------------------- |
| `data_pipeline_dag.py`              | ETL pipeline (Bronze → Silver → Gold)     |
| `scheduled_training_dag.py`         | Scheduled model training & MLflow logging |
| `daily_inference_monitoring_dag.py` | Daily inference + model monitoring        |
  

Steps:  
	1.	Open Airflow UI (http://localhost:8080)  
	2.	Trigger the DAG manually or let it run on schedule  

### Option B – via Python Scripts
Run specific stages manually:

Or run each script: 


```bash
python data_processing_pipeline.py 
```
Creates Bronze → Silver → Gold tables (full ETL)
```bash
python model_training_pipeline.py 2024-12-01
```
Trains the model for a chosen date and logs results to MLflow.
```bash
python nference_and_monitoring_pipeline.py
``` 
Runs inference and performs drift + performance monitoring.



