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
    """Apply silver transformations to attributes data - FIXED VERSION"""
    df_clean = df
    
    # STEP 1: First handle Age as string to remove underscores and validate
    df_clean = df_clean.withColumn(
        "Age_clean",
        F.regexp_replace(F.col("Age").cast("string"), "_", "")  # remove underscores
    )
    
    # STEP 2: Only cast to integer if it's a valid number
    df_clean = df_clean.withColumn(
        "Age_temp",
        F.when(
            F.col("Age_clean").rlike("^\\d+$") &  # Only contains digits
            (F.length(F.col("Age_clean")) > 0),   # Not empty
            F.col("Age_clean").cast(IntegerType())
        ).otherwise(None)
    )
    
    # STEP 3: Apply age validation rules
    df_clean = df_clean.withColumn("Age", 
        F.when(col("Age_temp") == -500, None)
        .when(col("Age_temp") > 100, None)
        .when(col("Age_temp") < 18, None)
        .otherwise(col("Age_temp"))
    )
    
    # STEP 4: Drop temporary columns
    df_clean = df_clean.drop("Age_clean", "Age_temp")
    
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
    """Apply silver transformations to financials data - COMPLETELY FIXED"""
    df_clean = df
    
    # Define numeric columns EXCLUDING Credit_History_Age (needs special handling)
    numeric_columns = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly", 
        "Monthly_Balance"
    ]

    # STEP 1: Clean regular numeric columns with SAFE casting
    for col_name in numeric_columns:
        if col_name in df_clean.columns:
            # Step 1a: Remove underscores and handle empty strings
            clean_string_col = F.regexp_replace(col(col_name).cast(StringType()), "_", "")
            
            # Step 1b: Use try_cast to safely convert to float (returns NULL for invalid values)
            df_clean = df_clean.withColumn(
                col_name + "_clean",
                F.expr(f"try_cast({col_name} as float)")
            )
            
            # Step 1c: Apply cleaning rules to the cleaned column
            if col_name == "Num_Bank_Accounts":
                df_clean = df_clean.withColumn(col_name + "_clean",
                    F.when((col(col_name + "_clean") == -1) | (col(col_name + "_clean") > 100), None)
                    .otherwise(col(col_name + "_clean"))
                )
            elif col_name == "Num_Credit_Card":
                df_clean = df_clean.withColumn(col_name + "_clean",
                    F.when(col(col_name + "_clean") > 100, None).otherwise(col(col_name + "_clean"))
                )
            
            # Step 1d: Replace original column and drop temp
            df_clean = df_clean.withColumn(col_name, col(col_name + "_clean"))
            df_clean = df_clean.drop(col_name + "_clean")

    # STEP 2: Handle Credit_History_Age separately - PROCESS AS STRING FIRST
    if "Credit_History_Age" in df_clean.columns:
        # First, ensure it's a string and clean it
        df_clean = df_clean.withColumn(
            "Credit_History_Age_string",
            F.coalesce(col("Credit_History_Age").cast(StringType()), F.lit(""))
        )
        
        # Extract years and months from the string
        df_clean = df_clean.withColumn(
            "Credit_History_Age_months",
            (
                F.coalesce(
                    F.regexp_extract(col("Credit_History_Age_string"), r"^(\d+)\s*Years", 1).cast(FloatType()), 
                    F.lit(0)
                ) * 12
                +
                F.coalesce(
                    F.regexp_extract(col("Credit_History_Age_string"), r"and\s*(\d+)\s*Months", 1).cast(FloatType()), 
                    F.lit(0)
                )
            )
        )
        # Replace the original column with the calculated months
        df_clean = df_clean.drop("Credit_History_Age", "Credit_History_Age_string")
        df_clean = df_clean.withColumnRenamed("Credit_History_Age_months", "Credit_History_Age")

    # STEP 3: Fill missing numeric values
    for col_name in numeric_columns:
        if col_name in df_clean.columns:
            df_clean = df_clean.fillna({col_name: 0.0})
    
    # Also fill Credit_History_Age if it exists
    if "Credit_History_Age" in df_clean.columns:
        df_clean = df_clean.fillna({"Credit_History_Age": 0.0})

    # STEP 4: Clean categorical columns
    categorical_fills = {
        "Credit_Mix": "Unknown",
        "Payment_of_Min_Amount": "Unknown", 
        "Payment_Behaviour": "Unknown",
        "Type_of_Loan": "Unknown"
    }
    
    for col_name, fill_value in categorical_fills.items():
        if col_name in df_clean.columns:
            df_clean = df_clean.fillna({col_name: fill_value})

    # Final cleanup
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
    
    # UPDATED: Calculate Days Past Due (DPD) - FIXED ORDER
    # Calculate monthly payment FIRST
    df_clean = df_clean.withColumn("monthly_payment", 
        F.when((col("loan_amt") > 0) & (col("tenure") > 0), 
               col("loan_amt") / col("tenure"))
        .otherwise(0)
    )
    
    # THEN calculate DPD using the monthly_payment column
    df_clean = df_clean.withColumn("dpd", 
        F.when(col("monthly_payment") > 0, 
               (col("overdue_amt") / col("monthly_payment")) * 30)
        .otherwise(0)
    )
    
    # Ensure DPD is non-negative and round to whole days
    df_clean = df_clean.withColumn("dpd",
        F.when(col("dpd") < 0, 0)
        .otherwise(F.round(col("dpd")))
        .cast(IntegerType())
    )
    
    # Handle missing values
    df_clean = df_clean.fillna({
        "loan_amt": 0,
        "due_amt": 0,
        "paid_amt": 0,
        "overdue_amt": 0,
        "balance": 0,
        "tenure": 0,
        "monthly_payment": 0
    })
    
    # Remove rows with missing critical fields
    df_clean = df_clean.filter(
        col("Customer_ID").isNotNull() & 
        col("loan_id").isNotNull() &
        col("loan_start_date").isNotNull()
    )
    
    return df_clean


def process_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark, file_naming="overwrite"):
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
        
        # FIX: Load from table-specific folder
        table_bronze_folder = os.path.join(bronze_directory, table_name)
        filepath = os.path.join(table_bronze_folder, table_config['file_pattern'])
        
        # Load with inferSchema to get the raw data
        df = spark.read.csv(filepath, header=True, inferSchema=True)
        print(f'Loaded from: {filepath}, row count: {df.count()}')
        
        # FIX: APPLY TRANSFORMATIONS FIRST, then enforce schema
        df = table_config['transformations'](df)
        
        # FIX: Now enforce the target schema with safe casting
        for column, new_type in table_config['schema_map'].items():
            if column in df.columns:
                if isinstance(new_type, IntegerType):
                    # Use try_cast for integers to handle malformed data
                    df = df.withColumn(column, F.expr(f"try_cast({column} as int)"))
                elif isinstance(new_type, FloatType):
                    # Use try_cast for floats to handle malformed data
                    df = df.withColumn(column, F.expr(f"try_cast({column} as float)"))
                else:
                    # For other types, use regular cast
                    df = df.withColumn(column, col(column).cast(new_type))

        silver_tables[table_name] = df
        print(f"Silver table {table_name} processed: {df.count()} rows")

    # Save silver tables
    for table_name, df in silver_tables.items():
        # Create table-specific folder
        table_silver_folder = os.path.join(silver_directory, table_name)
        os.makedirs(table_silver_folder, exist_ok=True)
        
        if file_naming == "date_specific":
            # Create date-specific filename
            date_suffix = snapshot_date_str.replace("-", "_")
            output_filename = f"silver_{table_name}_{date_suffix}.parquet"
            output_path = os.path.join(table_silver_folder, output_filename)
            
            # Overwrite this specific date file
            df.write.mode("overwrite").parquet(output_path)
            print(f'Saved {table_name} for {snapshot_date_str} to: {output_path}')
        else:
            # Original behavior - overwrite entire folder
            df.write.mode("overwrite").parquet(table_silver_folder)
            print(f'Saved {table_name} to: {table_silver_folder}')
    
    return silver_tables

