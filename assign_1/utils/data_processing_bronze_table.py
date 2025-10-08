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


def process_bronze_table(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    data_sources = {
            'clickstream': "data/feature_clickstream.csv",
            'attributes': "data/features_attributes.csv", 
            'financials': "data/features_financials.csv",
            'loan_daily': "data/lms_loan_daily.csv"
        }
    bronze_tables = {}

    # load data - IRL ingest from back end source system
    for table_name, file_path in data_sources.items():
        print(f"Processing {table_name}...")
        
        # Load and filter data
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        
        # Apply snapshot date filter (adjust column name as needed)
        if 'snapshot_date' in df.columns:
            df = df.filter(col('snapshot_date') == snapshot_date)

        # save bronze table to datamart - IRL connect to database to write
        partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-','_')}.csv"
        filepath = os.path.join(bronze_lms_directory, partition_name)
        df.toPandas().to_csv(filepath, index=False)
        print(f'Saved {table_name} to: {filepath}')

        bronze_tables[table_name] = df

    return bronze_tables
