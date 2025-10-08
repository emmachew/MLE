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

def process_gold_tables(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, gold_feature_store_directory, spark, dpd_threshold=1, mob_threshold=6):
    
    # Load all silver tables
    silver_tables = {}
    table_names = ['loan_daily', 'attributes', 'financials', 'clickstream']
    
    for table_name in table_names:
        partition_name = f"silver_{table_name}_{snapshot_date_str.replace('-','_')}.parquet"
        filepath = os.path.join(silver_loan_daily_directory, partition_name)
        try:
            df = spark.read.parquet(filepath)
            silver_tables[table_name] = df
            print(f"✓ Loaded {table_name}: {df.count()} rows")
        except Exception as e:
            print(f"✗ Failed to load {table_name}: {e}")
            silver_tables[table_name] = None

    # 1. Create Label Store
    print("\n1. CREATING LABEL STORE")
    label_store_df = create_label_store(silver_tables['loan_daily'], dpd_threshold, mob_threshold, snapshot_date_str)
    
    # Save label store
    label_partition_name = f"gold_label_store_{snapshot_date_str.replace('-','_')}.parquet"
    label_filepath = os.path.join(gold_label_store_directory, label_partition_name)
    label_store_df.write.mode("overwrite").parquet(label_filepath)
    print(f"✓ Label store saved: {label_filepath}")

    # 2. Create Feature Store
    print("\n2. CREATING FEATURE STORE")
    feature_store_df = create_feature_store(silver_tables, label_store_df, snapshot_date_str)
    
    # Save feature store
    feature_partition_name = f"gold_feature_store_{snapshot_date_str.replace('-','_')}.parquet"
    feature_filepath = os.path.join(gold_label_store_directory, feature_partition_name)
    feature_store_df.write.mode("overwrite").parquet(feature_filepath)
    print(f"✓ Feature store saved: {feature_filepath}")

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
    
    # Add label definition
    df_label = df_label.withColumn("label_definition", 
        F.lit(f"DPD_{dpd_threshold}+_MOB_{mob_threshold}+").cast(StringType())
    )
    
    # Add snapshot metadata
    df_label = df_label.withColumn("label_snapshot_date", 
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

def create_feature_store(silver_tables, label_store_df, snapshot_date_str):
    """Create feature store by combining features from all silver tables"""
    
    print("Building feature store from all silver tables...")
    
    # Start with label store as base (contains Customer_ID and labels)
    feature_store_df = label_store_df.select("Customer_ID", "label", "label_definition")
    
    # 1. Add attributes features (already cleaned in silver)
    if silver_tables.get('attributes') is not None:
        print("  Adding attributes features...")
        attributes_features = silver_tables['attributes']
        
        # Select only the required attributes features
        attr_cols = ["Customer_ID", "Age", "Occupation"]
        existing_attr_cols = [col for col in attr_cols if col in attributes_features.columns]
        attributes_selected = attributes_features.select(*existing_attr_cols)
        
        feature_store_df = feature_store_df.join(attributes_selected, "Customer_ID", "left")
        print(f"    Added {len(existing_attr_cols) - 1} attributes features: {existing_attr_cols[1:]}")
    
    # 2. Add financials features (already cleaned in silver)
    if silver_tables.get('financials') is not None:
        print("  Adding financials features...")
        financials_features = silver_tables['financials']
        
        # Create Loan_Type binary features from Type_of_Loan (if it exists)
        if "Type_of_Loan" in financials_features.columns:
            # Create binary columns for common loan types
            loan_types = ["Home Loan", "Personal Loan", "Student Loan", "Auto Loan", "Business Loan"]
            for loan_type in loan_types:
                financials_features = financials_features.withColumn(
                    f"Loan_Type_{loan_type.replace(' ', '_')}",
                    when(col("Type_of_Loan").contains(loan_type), 1).otherwise(0)
                )
        
        # Select relevant financial features
        financial_cols = [
            "Customer_ID", 
            "Delay_from_due_date", 
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
        
        feature_store_df = feature_store_df.join(financials_selected, "Customer_ID", "left")
        print(f"    Added {len(existing_fin_cols) - 1} financials features")
    
    # 3. Add loan behavior features (aggregations from silver loan data)
    if silver_tables.get('loan_daily') is not None:
        print("  Adding loan behavior features...")
        loan_features = silver_tables['loan_daily']
        
        # Calculate loan-level features with window functions for std dev
        loan_features = loan_features.withColumn("loan_days_past_due_std", 
            F.stddev("dpd").over(Window.partitionBy("Customer_ID"))
        )
        
        # Customer-level loan aggregations
        loan_agg = loan_features.groupBy("Customer_ID").agg(
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
            
            # DPD aggregations (including the window-calculated std dev)
            F.first("loan_days_past_due_std").alias("loan_days_past_due_std"),
            F.mean("dpd").alias("dpd_mean"),
            F.max("dpd").alias("dpd_max")
        )
        
        feature_store_df = feature_store_df.join(loan_agg, "Customer_ID", "left")
        print(f"    Added {len(loan_agg.columns) - 1} loan behavior features")
    
    # 4. Add clickstream features (specific aggregations)
    if silver_tables.get('clickstream') is not None:
        print("  Adding clickstream features...")
        clickstream_features = silver_tables['clickstream']
        
        # If clickstream data needs aggregation (multiple records per customer)
        customer_record_counts = clickstream_features.groupBy("Customer_ID").count()
        if customer_record_counts.filter(col("count") > 1).count() > 0:
            # Aggregate to customer level with specified features
            clickstream_agg = clickstream_features.groupBy("Customer_ID").agg(
                F.mean("fe_5").alias("fe_5_mean"),
                F.mean("fe_10").alias("fe_10_mean"),
                F.min("fe_5").alias("fe_5_min"),
                F.min("fe_9").alias("fe_9_min"),
                F.min("fe_4").alias("fe_4_min")
            )
            clickstream_selected = clickstream_agg
        else:
            # Data is already at customer level, just select the required columns
            clickstream_cols = [
                "Customer_ID", "fe_5", "fe_10", "fe_9", "fe_4"
            ]
            existing_click_cols = [col for col in clickstream_cols if col in clickstream_features.columns]
            clickstream_temp = clickstream_features.select(*existing_click_cols)
            
            # Rename columns to match the expected feature names
            column_mapping = {
                "fe_5": "fe_5_mean",
                "fe_10": "fe_10_mean", 
                "fe_5": "fe_5_min",
                "fe_9": "fe_9_min",
                "fe_4": "fe_4_min"
            }
            for old_col, new_col in column_mapping.items():
                if old_col in clickstream_temp.columns:
                    clickstream_temp = clickstream_temp.withColumnRenamed(old_col, new_col)
            
            clickstream_selected = clickstream_temp
        
        feature_store_df = feature_store_df.join(clickstream_selected, "Customer_ID", "left")
        print(f"    Added clickstream features: fe_5_mean, fe_10_mean, fe_5_min, fe_9_min, fe_4_min")
    
    # Fill any remaining missing values
    print("  Handling missing values...")
    for column in feature_store_df.columns:
        if column not in ["Customer_ID", "label", "label_definition"]:
            if feature_store_df.schema[column].dataType in [IntegerType(), FloatType(), DoubleType()]:
                # Fill numeric missing values with 0 (since data is already cleaned)
                feature_store_df = feature_store_df.fillna({column: 0})
            else:
                # Fill categorical missing values with appropriate defaults
                if column == "Occupation":
                    feature_store_df = feature_store_df.fillna({column: "Unknown"})
                else:
                    feature_store_df = feature_store_df.fillna({column: "Unknown"})
    
    # Add snapshot metadata
    feature_store_df = feature_store_df.withColumn("feature_snapshot_date", 
        F.lit(snapshot_date_str).cast(StringType())
    )
    
    # Final feature count
    feature_columns = [col for col in feature_store_df.columns if col not in 
                      ["Customer_ID", "label", "label_definition", "feature_snapshot_date"]]
    
    print(f"Final feature store: {feature_store_df.count()} customers, {len(feature_columns)} features")
    print(f"Feature columns: {feature_columns}")
    
    return feature_store_df