import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType
from pyspark.sql.window import Window

def process_gold_tables(snapshot_date_str, silver_directory, gold_label_store_directory, gold_feature_store_directory, spark, dpd_threshold=30, mob_threshold=6, file_naming="date_specific"):
    
    # Load all silver tables using partition filtering
    silver_tables = {}
    table_names = ['loan_daily', 'attributes', 'financials', 'clickstream']
    
    for table_name in table_names:
        table_silver_folder = os.path.join(silver_directory, table_name)
        try:
            if file_naming == "date_specific":
                # Read specific date file for this processing run
                date_suffix = snapshot_date_str.replace("-", "_")
                specific_file = f"silver_{table_name}_{date_suffix}.parquet"
                specific_file_path = os.path.join(table_silver_folder, specific_file)
                
                print(f"Looking for: {specific_file_path}")
                if os.path.exists(specific_file_path):
                    df = spark.read.parquet(specific_file_path)
                    silver_tables[table_name] = df
                    print(f"✓ Loaded {table_name} for {snapshot_date_str}: {df.count()} rows")
                else:
                    print(f"✗ No data found for {table_name} on {snapshot_date_str}")
                    print(f"  Expected file: {specific_file_path}")
                    silver_tables[table_name] = None
            else:
                # Original behavior - read entire folder (for partitioned data)
                df = spark.read.parquet(table_silver_folder)
                silver_tables[table_name] = df
                print(f"✓ Loaded {table_name}: {df.count()} rows from {table_silver_folder}")
                
        except Exception as e:
            print(f"✗ Failed to load {table_name} from {table_silver_folder}: {e}")
            silver_tables[table_name] = None

    # Check if we have any data to process
    available_tables = [name for name, df in silver_tables.items() if df is not None and df.count() > 0]
    if not available_tables:
        print(f"⚠️ No data available for {snapshot_date_str}. Skipping gold processing.")
        return {'label_store': None, 'feature_store': None}

    print(f"Available tables with data: {available_tables}")

    # 1. Create Label Store
    print("\n1. CREATING LABEL STORE")
    try:
        label_store_df = create_label_store(silver_tables['loan_daily'], dpd_threshold, mob_threshold, snapshot_date_str)
    except Exception as e:
        print(f"✗ Failed to create label store: {e}")
        label_store_df = None

    # Save label store
    if label_store_df is not None and label_store_df.count() > 0:
        os.makedirs(gold_label_store_directory, exist_ok=True)
        
        if file_naming == "date_specific":
            # Create date-specific filename for labels
            date_suffix = snapshot_date_str.replace("-", "_")
            label_output_filename = f"gold_labels_{date_suffix}.parquet"
            label_output_path = os.path.join(gold_label_store_directory, label_output_filename)
            
            label_store_df.write.mode("overwrite").parquet(label_output_path)
            print(f"✓ Label store saved for {snapshot_date_str} to: {label_output_path}")
        else:
            # Original behavior - overwrite entire folder
            label_store_df.write.mode("overwrite").parquet(gold_label_store_directory)
            print(f"✓ Label store saved to: {gold_label_store_directory}")
    else:
        print("⚠️ No label data to save")

    # 2. Create Feature Store
    print("\n2. CREATING FEATURE STORE")
    try:
        feature_store_df = create_feature_store(silver_tables, snapshot_date_str)
        if feature_store_df is not None:
            feature_store_df = feature_store_df.dropDuplicates(["Customer_ID", "snapshot_date"])
    except Exception as e:
        print(f"✗ Failed to create feature store: {e}")
        feature_store_df = None

    # Save feature store
    if feature_store_df is not None and feature_store_df.count() > 0:
        os.makedirs(gold_feature_store_directory, exist_ok=True)
        
        if file_naming == "date_specific":
            # Create date-specific filename for features
            date_suffix = snapshot_date_str.replace("-", "_")
            feature_output_filename = f"gold_features_{date_suffix}.parquet"
            feature_output_path = os.path.join(gold_feature_store_directory, feature_output_filename)
            
            feature_store_df.write.mode("overwrite").parquet(feature_output_path)
            print(f"✓ Feature store saved for {snapshot_date_str} to: {feature_output_path}")
        else:
            # Original behavior - overwrite entire folder
            feature_store_df.write.mode("overwrite").parquet(gold_feature_store_directory)
            print(f"✓ Feature store saved to: {gold_feature_store_directory}")
    else:
        print("⚠️ No feature data to save")

    return {
        'label_store': label_store_df,
        'feature_store': feature_store_df
    }

def create_label_store(loan_silver_df, dpd_threshold, mob_threshold, snapshot_date_str):
    """Create label store with default labels"""
    
    if loan_silver_df is None:
        raise ValueError("Loan silver table not available for label creation")
    
    print(f"Creating labels: DPD ≥ {dpd_threshold} at MOB ≥ {mob_threshold}")
    
    # Filter for stable MOB period
    df_label = loan_silver_df.filter(col("mob") >= mob_threshold)
    
    # Create default label
    df_label = df_label.withColumn("label", 
        when(col("dpd") >= dpd_threshold, 1).otherwise(0).cast(IntegerType())
    )
    
    # Add label definition and snapshot metadata
    df_label = df_label.withColumn("label_definition", 
        F.lit(f"DPD_{dpd_threshold}+_MOB_{mob_threshold}+").cast(StringType())
    ).withColumn("label_snapshot_date", 
        F.lit(snapshot_date_str).cast(StringType())
    )
    
    # Select final columns
    label_cols = [
        "loan_id", "Customer_ID", "label", "label_definition", 
        "label_snapshot_date", "snapshot_date", "mob", "dpd"
    ]
    
    # Only include columns that exist
    existing_cols = [col for col in label_cols if col in df_label.columns]
    df_label = df_label.select(*existing_cols)
    
    print(f"Label distribution:")
    label_counts = df_label.groupBy("label").count().collect()
    for row in label_counts:
        print(f"  Label {row['label']}: {row['count']} records")
    
    return df_label

def create_feature_store(silver_tables, snapshot_date_str):
    """Create feature store by combining features from all silver tables - NO LABELS"""
    
    print("Building feature store from all silver tables...")
    
    # Get unique Customer_IDs and snapshot_dates from loan data
    if silver_tables.get('loan_daily') is not None:
        customer_base = silver_tables['loan_daily'].select("Customer_ID", "snapshot_date").distinct()
    else:
        # Fallback: get from any available table
        for table_name, table_df in silver_tables.items():
            if table_df is not None and "Customer_ID" in table_df.columns and "snapshot_date" in table_df.columns:
                customer_base = table_df.select("Customer_ID", "snapshot_date").distinct()
                break
        else:
            raise ValueError("No silver tables available with Customer_ID and snapshot_date")
    
    feature_store_df = customer_base
    
    # Add snapshot metadata
    feature_store_df = feature_store_df.withColumn("feature_snapshot_date", 
        F.lit(snapshot_date_str).cast(StringType())
    )
    
    # 1. Add attributes features
    if silver_tables.get('attributes') is not None:
        print("  Adding attributes features...")
        attributes_features = silver_tables['attributes']
        
        # Select only the required attributes features
        attr_cols = ["Customer_ID", "snapshot_date", "Age", "Occupation"]
        existing_attr_cols = [col for col in attr_cols if col in attributes_features.columns]
        attributes_selected = attributes_features.select(*existing_attr_cols)
        
        feature_store_df = feature_store_df.join(
            attributes_selected, 
            ["Customer_ID", "snapshot_date"], 
            "left"
        )
        print(f"    Added {len(existing_attr_cols) - 2} attributes features")
    
    # 2. Add financials features
    if silver_tables.get('financials') is not None:
        print("  Adding financials features...")
        financials_features = silver_tables['financials']
        
        # Create Loan_Type binary features from Type_of_Loan (if it exists)
        if "Type_of_Loan" in financials_features.columns:
            loan_types = [
                "Home Loan", "Personal Loan", "Student Loan", "Auto Loan", "Business Loan",
                "Credit-Builder Loan", "Home Equity Loan", "Debt Consolidation Loan",
                "Mortgage Loan", "Not Specified", "Payday Loan"
            ]
            for loan_type in loan_types:
                financials_features = financials_features.withColumn(
                    f"Loan_Type_{loan_type.replace(' ', '_')}",
                    when(col("Type_of_Loan").contains(loan_type), 1).otherwise(0)
                )
        
        financial_cols = [
            "Customer_ID", "snapshot_date",
            "Delay_from_due_date", 
            "Outstanding_Debt",
            "Amount_invested_monthly", 
            "Interest_Rate",
            "Num_Bank_Accounts", 
            "Num_Credit_Card"
        ]
        
        # Add loan type columns
        loan_type_cols = [col for col in financials_features.columns if col.startswith("Loan_Type_")]
        financial_cols.extend(loan_type_cols)
        
        existing_fin_cols = [col for col in financial_cols if col in financials_features.columns]
        financials_selected = financials_features.select(*existing_fin_cols)
        
        feature_store_df = feature_store_df.join(
            financials_selected, 
            ["Customer_ID", "snapshot_date"], 
            "left"
        )
        print(f"    Added {len(existing_fin_cols) - 2} financials features")
    
    # 3. Add loan behavior features
    if silver_tables.get('loan_daily') is not None:
        print("  Adding loan behavior features...")
        loan_features = silver_tables['loan_daily']
        
        # Customer-level loan aggregations by snapshot_date
        loan_agg = loan_features.groupBy("Customer_ID", "snapshot_date").agg(
            # Loan amount aggregations
            F.sum("loan_amt").alias("Loan_amt_sum"),
            F.mean("loan_amt").alias("Loan_amt_mean"),
            F.stddev("loan_amt").alias("Loan_amt_std"),
            
            # Tenure aggregations
            F.mean("tenure").alias("Loan_tenure_mean"),
            F.max("tenure").alias("Loan_tenure_max"),
            
            # Overdue amount aggregations
            F.sum("overdue_amt").alias("Loan_overdue_amt_sum"),
            F.mean("overdue_amt").alias("Loan_overdue_amt_mean"),
            F.max("overdue_amt").alias("Loan_overdue_amt_max"),
            
            # Balance aggregations
            F.sum("balance").alias("Loan_balance_sum"),
            F.mean("balance").alias("Loan_balance_mean"),
            
            # DPD aggregations
            F.mean("dpd").alias("dpd_mean"),
            F.max("dpd").alias("dpd_max"),
            F.count("loan_id").alias("loan_count")
        )
        
        feature_store_df = feature_store_df.join(
            loan_agg, 
            ["Customer_ID", "snapshot_date"], 
            "left"
        )
        print(f"    Added {len(loan_agg.columns) - 2} loan behavior features")
    
    # 4. Add clickstream features
    if silver_tables.get('clickstream') is not None:
        print("  Adding clickstream features...")
        clickstream_df = silver_tables['clickstream']
        
        # Simple aggregation by Customer_ID and snapshot_date
        clickstream_agg = clickstream_df.groupBy("Customer_ID", "snapshot_date").agg(
            F.count("*").alias("clickstream_total_events"),
            F.mean("fe_5").alias("clickstream_fe_5_mean"),
            F.sum("fe_5").alias("clickstream_fe_5_sum"),
            F.stddev("fe_5").alias("clickstream_fe_5_std"),
            F.mean("fe_9").alias("clickstream_fe_9_mean"),
            F.min("fe_9").alias("clickstream_fe_9_min"),
            F.mean("fe_4").alias("clickstream_fe_4_mean"),
            F.min("fe_4").alias("clickstream_fe_4_min"),
            F.mean("fe_10").alias("clickstream_fe_10_mean"),
            F.min("fe_10").alias("clickstream_fe_10_min")
        )
        
        feature_store_df = feature_store_df.join(
            clickstream_agg, 
            ["Customer_ID", "snapshot_date"], 
            "left"
        )
        print(f"    Added {len(clickstream_agg.columns) - 2} clickstream features")
    
    # Fill any remaining missing values
    print("  Handling missing values...")
    for column in feature_store_df.columns:
        if column not in ["Customer_ID", "snapshot_date", "feature_snapshot_date"]:
            if feature_store_df.schema[column].dataType in [IntegerType(), FloatType(), DoubleType()]:
                feature_store_df = feature_store_df.fillna({column: 0})
            elif column == "Occupation":
                feature_store_df = feature_store_df.fillna({column: "Unknown"})
    
    # Final feature count
    feature_columns = [col for col in feature_store_df.columns if col not in 
                      ["Customer_ID", "snapshot_date", "feature_snapshot_date"]]
    
    print(f"Final feature store: {feature_store_df.count()} records, {len(feature_columns)} features")
    print(f"Feature columns: {feature_columns}")
    
    return feature_store_df