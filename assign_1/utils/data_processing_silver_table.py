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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze tables
    bronze_tables = {
        'clickstream': {
            'file_pattern': "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv',
            'schema_map': {
                "fe_1": IntegerType(),
                "fe_2": IntegerType(),
                "fe_3": IntegerType(),
                "fe_4": IntegerType(),
                "fe_5": IntegerType(),
                "fe_6": IntegerType(),
                "fe_7": IntegerType(),
                "fe_8": IntegerType(),
                "fe_9": IntegerType(),
                "fe_10": IntegerType(),
                "fe_11": IntegerType(),
                "fe_12": IntegerType(),
                "fe_13": IntegerType(),
                "fe_14": IntegerType(),
                "fe_15": IntegerType(),
                "fe_16": IntegerType(),
                "fe_17": IntegerType(),
                "fe_18": IntegerType(),
                "fe_19": IntegerType(),
                "fe_20": IntegerType(),
                "Customer_ID": StringType(),
                "snapshot_date": DateType(),
            },
            'transformations': lambda df: apply_clickstream_silver_transformations(df)
        },
        'attributes': {
            'file_pattern': "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv',
            'schema_map': {
                "Customer_ID": StringType(),
                "Name": StringType(),
                "Age": IntegerType(),
                "SSN": StringType(),
                "Occupation": StringType(),
                "snapshot_date": DateType(),
            },
            'transformations': lambda df: apply_attributes_silver_transformations(df)
        },
        'financials': {
            'file_pattern': "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv',
            'schema_map': {
                "Customer_ID": StringType(),
                "Annual_Income": FloatType(),
                "Monthly_Inhand_Salary": FloatType(),
                "Num_Bank_Accounts": IntegerType(),
                "Num_Credit_Card": IntegerType(),
                "Interest_Rate": FloatType(),
                "Num_of_Loan": IntegerType(),
                "Type_of_Loan": StringType(),
                "Delay_from_due_date": IntegerType(),
                "Num_of_Delayed_Payment": IntegerType(),
                "Changed_Credit_Limit": FloatType(),
                "Num_Credit_Inquiries": IntegerType(),
                "Credit_Mix": StringType(),
                "Outstanding_Debt": FloatType(),
                "Credit_Utilization_Ratio": FloatType(),
                "Credit_History_Age": FloatType(),
                "Payment_of_Min_Amount": StringType(),
                "Total_EMI_per_month": FloatType(),
                "Amount_invested_monthly": FloatType(),
                "Payment_Behaviour": StringType(),
                "Monthly_Balance": FloatType(),
                "snapshot_date": DateType(),
            },
            'transformations': lambda df: apply_financials_silver_transformations(df)
        },
        'loan_daily': {
            'file_pattern': "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv',
            'schema_map': {
                "loan_id": StringType(),
                "Customer_ID": StringType(),
                "loan_start_date": DateType(),
                "tenure": IntegerType(),
                "installment_num": IntegerType(),
                "loan_amt": FloatType(),
                "due_amt": FloatType(),
                "paid_amt": FloatType(),
                "overdue_amt": FloatType(),
                "balance": FloatType(),
                "snapshot_date": DateType(),
            },
            'transformations': lambda df: apply_loan_silver_transformations(df)
        }
    }
    
    silver_tables = {}
    
    # Process each bronze table
    for table_name, table_config in bronze_tables.items():
        print(f"\nProcessing silver table: {table_name}")
        
        # Load bronze table
        filepath = os.path.join(bronze_lms_directory, table_config['file_pattern'])
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print(f'Loaded from: {filepath}, row count: {df.count()}')
        
        # Clean data: enforce schema
        for column, new_type in table_config['schema_map'].items():
            if column in df.columns:
                df = df.withColumn(column, col(column).cast(new_type))
        
        # Apply silver transformations
        df = table_config['transformations'](df)

        silver_tables[table_name] = df
        print(f"Silver table {table_name} processed: {df.count()} rows")

    # Save silver tables - IRL connect to database to write
    for table_name, df in silver_tables.items():
        partition_name = f"silver_{table_name}_{snapshot_date_str.replace('-','_')}.parquet"
        filepath = os.path.join(silver_loan_daily_directory, partition_name)
        df.write.mode("overwrite").parquet(filepath)
        print(f'Saved {table_name} to: {filepath}')
    
    return silver_tables

    # Silver transformation functions
def apply_clickstream_silver_transformations(df):
    """Apply silver transformations to clickstream data"""
    df_clean = df
    
    # Handle missing values in clickstream features
    for i in range(1, 21):
        col_name = f"fe_{i}"
        if col_name in df_clean.columns:
            # Fill missing values with 0 (assuming no activity)
            df_clean = df_clean.fillna({col_name: 0})
    
    # Ensure Customer_ID has no missing values
    df_clean = df_clean.filter(col("Customer_ID").isNotNull())
    
    return df_clean

def apply_attributes_silver_transformations(df):
    """Apply silver transformations to attributes data"""
    df_clean = df
    
    # Age cleaning
    df_clean = df_clean.withColumn("Age", 
        F.when(col("Age") == -500, None)
        .when(col("Age") > 100, None)
        .when(col("Age") < 18, None)
        .otherwise(col("Age"))
    )
    
    # Remove underscores from Age (if stored as string with underscores)
    df_clean = df_clean.withColumn("Age", 
        F.regexp_replace(col("Age").cast(StringType()), "_", "").cast(IntegerType())
    )
    
    # SSN cleaning
    df_clean = df_clean.withColumn("SSN",
        F.when(col("SSN").contains("#F%$D@*&8"), None)
        .otherwise(col("SSN"))
    )
    
    # Occupation cleaning
    df_clean = df_clean.withColumn("Occupation",
        F.when(col("Occupation").contains("_______"), None)
        .otherwise(col("Occupation"))
    )
    
    # Fill missing values
    df_clean = df_clean.fillna({
        "Age": 0,  # Will be handled in gold layer
        "Occupation": "Unknown"
    })
    
    # Remove rows with missing Customer_ID
    df_clean = df_clean.filter(col("Customer_ID").isNotNull())
    
    return df_clean 

def apply_financials_silver_transformations(df):
    """Apply silver transformations to financials data"""
    df_clean = df
    
    numeric_columns = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Credit_History_Age", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance"
    ]

    # 1. Clean Numeric Columns (Remove underscores, cast, and apply specific rules)
    for col_name in numeric_columns:
        if col_name in df_clean.columns:
            # First: Remove underscores and attempt conversion to FloatType
            # This turns non-convertible strings into NULLs.
            df_clean = df_clean.withColumn(col_name,
                F.regexp_replace(col(col_name).cast(StringType()), "_", "").cast(FloatType())
            )
            
            # Second: Apply specific cleaning rules (which may also set values to NULL)
            if col_name == "Num_Bank_Accounts":
                df_clean = df_clean.withColumn(col_name,
                    F.when((col(col_name) == -1) | (col(col_name) > 100), None)
                    .otherwise(col(col_name))
                )
            # ... (other cleaning rules for Num_Credit_Card, Interest_Rate, etc. go here) ...
            elif col_name == "Num_Credit_Card":
                df_clean = df_clean.withColumn(col_name, F.when(col(col_name) > 100, None).otherwise(col(col_name)))
            elif col_name == "Interest_Rate":
                df_clean = df_clean.withColumn(col_name, F.when(col(col_name) > 100, None).otherwise(col(col_name)))
            elif col_name == "Num_of_Loan":
                df_clean = df_clean.withColumn(col_name, F.when((col(col_name) == -1) | (col(col_name) > 100), None).otherwise(col(col_name)))
            elif col_name in ["Delay_from_due_date", "Num_of_Delayed_Payment"]:
                df_clean = df_clean.withColumn(col_name, F.when(col(col_name) < 0, None).otherwise(col(col_name)))

    # 2. Clean Categorical Columns (unchanged from your code)
    if "Credit_Mix" in df_clean.columns:
        df_clean = df_clean.withColumn("Credit_Mix",
            F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix"))
        )
    if "Payment_Behaviour" in df_clean.columns:
        df_clean = df_clean.withColumn("Payment_Behaviour",
            F.when(col("Payment_Behaviour").contains("!@9#%8"), "Unknown").otherwise(col("Payment_Behaviour"))
        )

    # 3. Impute Missing Values (Calculated median for numeric, hard-coded for categorical)
    
    # Calculate and fill numeric NaNs with median (THIS IS WHERE THE FIX IS APPLIED)
    for col_name in numeric_columns:
        if col_name in df_clean.columns:
            # Filter non-nulls only for the calculation
            df_non_null = df_clean.filter(col(col_name).isNotNull())
            
            # Calculate median only if there is data left to calculate on
            if df_non_null.count() > 0:
                try:
                    median_value = df_non_null.approxQuantile(col_name, [0.5], 0.01)[0]
                except Exception as e:
                    # Fallback if approxQuantile still fails (e.g., all values are NaN after cleaning)
                    print(f"Warning: Failed to calculate median for {col_name}. Error: {e}")
                    median_value = 0.0 # Safe fallback
            else:
                median_value = 0.0 # Safe fallback if no data is left

            df_clean = df_clean.fillna({col_name: median_value})
    
    # Fill missing categorical values (unchanged from your code)
    df_clean = df_clean.fillna({
        "Credit_Mix": "Unknown",
        "Payment_of_Min_Amount": "Unknown",
        "Payment_Behaviour": "Unknown",
        "Type_of_Loan": "Unknown"
    })
    
    # Final cleanup (unchanged)
    df_clean = df_clean.filter(col("Customer_ID").isNotNull())
    
    return df_clean

def apply_loan_silver_transformations(df):
    """Apply silver transformations to loan data"""
    df_clean = df
    
    # Convert dates
    date_columns = ["loan_start_date", "snapshot_date"]
    for col_name in date_columns:
        if col_name in df_clean.columns:
            df_clean = df_clean.withColumn(col_name, col(col_name).cast(DateType()))
    
    # Calculate Month on Book (MOB)
    df_clean = df_clean.withColumn("mob", 
        F.round(F.datediff(col("snapshot_date"), col("loan_start_date")) / 30).cast(IntegerType())
    )
    
    # Filter out negative MOB values (data issues)
    df_clean = df_clean.filter(col("mob") >= 0)
    
    # Calculate Days Past Due (DPD)
    df_clean = df_clean.withColumn("installments_missed", 
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
    ).fillna(0)
    
    df_clean = df_clean.withColumn("first_missed_date", 
        F.when(col("installments_missed") > 0, 
             F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
        .otherwise(col("snapshot_date"))
        .cast(DateType())
    )
    
    df_clean = df_clean.withColumn("dpd", 
        F.when(col("overdue_amt") > 0.0, 
             F.datediff(col("snapshot_date"), col("first_missed_date")))
        .otherwise(0)
        .cast(IntegerType())
    )
    
    # Handle missing values
    df_clean = df_clean.fillna({
        "loan_amt": 0,
        "due_amt": 0,
        "paid_amt": 0,
        "overdue_amt": 0,
        "balance": 0,
        "tenure": 0
    })
    
    # Remove rows with missing critical fields
    df_clean = df_clean.filter(
        col("Customer_ID").isNotNull() & 
        col("loan_id").isNotNull() &
        col("loan_start_date").isNotNull()
    )
    
    return df_clean
