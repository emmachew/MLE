#!/usr/bin/env python3
# model_inference.py

import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import sys

from pyspark.sql.functions import col, to_date
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main(snapshot_date_str, model_name):
    """
    Main inference function that takes a snapshot date and model name,
    runs inference, and saves predictions to datamart
    """
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # Set up config
    config = {}
    config["snapshot_date_str"] = snapshot_date_str
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = model_name
    config["model_bank_directory"] = "/app/model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]

    print("=== CONFIG ===")
    pprint.pprint(config)

    # Load model artefact from model bank
    try:
        with open(config["model_artefact_filepath"], 'rb') as file:
            model_artefact = pickle.load(file)
        print("✅ Model loaded successfully! " + config["model_artefact_filepath"])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        spark.stop()
        return

    # Load feature store
    try:
        folder_path = "/app/datamart/gold/feature_store/"
        files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
        feature_sdf = spark.read.option("header", "true").parquet(*files_list)

        # Ensure snapshot_date is in proper DateType
        feature_sdf = feature_sdf.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

        print(f"📊 Total row_count in feature store: {feature_sdf.count()}")

        # Extract features for specific snapshot date
        feature_sdf = feature_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
        print(f"🎯 Extracted features_sdf: {feature_sdf.count()} for date: {config['snapshot_date']}")

        if feature_sdf.count() == 0:
            print(f"⚠️  No features found for date: {config['snapshot_date']}")
            spark.stop()
            return

        feature_pdf = feature_sdf.toPandas()
    except Exception as e:
        print(f"❌ Failed to load feature store: {e}")
        spark.stop()
        return

    # Preprocess data for modeling
    try:
        # Define what columns are NOT features
        non_feature_cols = [
            'Customer_ID', 'snapshot_date', 'label', 'loan_id', 
            'label_definition', 'label_snapshot_date', 'feature_snapshot_date', 'Occupation',
            'mob', 'dpd', 'dpd_mean', 'dpd_max', 'Loan_overdue_amt_sum', 'Loan_overdue_amt_mean',
            'Loan_overdue_amt_max', 'Loan_amt_sum', 'Loan_amt_mean', 'Loan_amt_std',
            'Loan_balance_sum', 'Loan_balance_mean', 'loan_count', 'Delay_from_due_date',
            'clickstream_total_events', 'Age'
        ]

        # All other columns are features 
        feature_cols = [column for column in feature_sdf.columns if column not in non_feature_cols]

        X_inference = feature_sdf.select(feature_cols)

        scaler = model_artefact["preprocessing_transformers"]["stdscaler"]

        # Convert to pandas DataFrame
        X_inference_pdf = X_inference.toPandas()

        # Ensure only numeric data
        X_inference_pdf = X_inference_pdf.apply(pd.to_numeric, errors="raise")

        # Apply scaler
        X_inference_scaled = scaler.transform(X_inference_pdf)

        # Convert back to DataFrame
        X_inference_final = pd.DataFrame(X_inference_scaled, columns=X_inference_pdf.columns)

        print(f'🎯 X_inference rows: {X_inference_final.shape[0]}')
    except Exception as e:
        print(f"❌ Failed during preprocessing: {e}")
        spark.stop()
        return

    # Model prediction inference
    try:
        # Load model
        model = model_artefact["model"]

        # Predict model
        y_inference = model.predict_proba(X_inference_final)[:, 1]

        # Prepare output
        y_inference_pdf = feature_pdf[["Customer_ID","snapshot_date"]].copy()
        y_inference_pdf["model_name"] = config["model_name"]
        y_inference_pdf["model_predictions"] = y_inference
        
        print(f"✅ Generated predictions for {len(y_inference_pdf)} customers")
    except Exception as e:
        print(f"❌ Failed during prediction: {e}")
        spark.stop()
        return

    # Save model inference to datamart gold table - FIXED FILE WRITING
    try:
        # Create directory
        gold_directory = f'/app/datamart/gold/model_predictions/{config["model_name"][:-4]}/'
        print(f"📁 Output directory: {gold_directory}")

        if not os.path.exists(gold_directory):
            os.makedirs(gold_directory)

        # Save gold table - FIXED: Use proper file path handling
        partition_name = config["model_name"][:-4] + "_predictions_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = os.path.join(gold_directory, partition_name)  # Use os.path.join for proper path
        
        print(f"💾 Attempting to save to: {filepath}")
        
        # Debug: Check if path already exists and what type it is
        if os.path.exists(filepath):
            if os.path.isdir(filepath):
                print(f"⚠️  Path exists as directory, removing: {filepath}")
                import shutil
                shutil.rmtree(filepath)
            else:
                print(f"⚠️  Path exists as file, removing: {filepath}")
                os.remove(filepath)
        
        # Convert to Spark DataFrame and write
        spark_df = spark.createDataFrame(y_inference_pdf)
        spark_df.write.mode("overwrite").parquet(filepath)
        
        # Verify the file was created
        if os.path.exists(filepath):
            print(f'✅ Successfully saved to: {filepath}')
            # Check if it's a directory (Spark creates directories for Parquet)
            if os.path.isdir(filepath):
                files_created = os.listdir(filepath)
                print(f"📁 Created Parquet directory with files: {files_created}")
        else:
            print(f"❌ File was not created at: {filepath}")
        
    except Exception as e:
        print(f"❌ Failed to save predictions: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        return

    print("🎉 Inference completed successfully!")
    spark.stop()

def backfill_inference(start_date_str, end_date_str, model_name):
    """
    Backfill inference for a range of dates
    """
    def generate_first_of_month_dates(start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        first_of_month_dates = []
        current_date = datetime(start_date.year, start_date.month, 1)

        while current_date <= end_date:
            first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        return first_of_month_dates

    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    
    print(f"🔄 Starting backfill from {start_date_str} to {end_date_str}")
    for i, snapshot_date in enumerate(dates_str_lst):
        print(f"📅 Processing {i+1}/{len(dates_str_lst)}: {snapshot_date}")
        main(snapshot_date, model_name)

if __name__ == "__main__":
    # Default values
    snapshot_date = "2024-12-01"
    model_name = "credit_model_2024_12_01.pkl"
    
    # Use command line arguments if provided
    if len(sys.argv) > 1:
        snapshot_date = sys.argv[1]
    if len(sys.argv) > 2:
        model_name = sys.argv[2]
    
    print(f"🚀 Starting inference for {snapshot_date} with model {model_name}")
    main(snapshot_date, model_name)