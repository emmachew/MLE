#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col, to_date, lit
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# In[35]:


# Build a .py script that takes a snapshot date, trains a model and outputs artefact into storage.


# ## set up pyspark session

# In[36]:


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# ## set up config

# In[37]:


# set up config
model_train_date_str = "2024-06-01"
train_test_period_months = 12
oot_period_months = 2
train_test_ratio = 0.8

config = {}
config["model_train_date_str"] = model_train_date_str
config["train_test_period_months"] = train_test_period_months
config["oot_period_months"] = oot_period_months
config["train_test_ratio"] = train_test_ratio

# Standardize all dates to datetime.date objects for consistency
config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d").date()

# For monthly data (1st of each month), use month boundaries instead of day subtraction
# Get the first day of the month for OOT period
config["oot_end_date"] = (config['model_train_date'].replace(day=1) - timedelta(days=1)).replace(day=1)
config["oot_start_date"] = (config["oot_end_date"] - relativedelta(months=oot_period_months - 1)).replace(day=1)

# Training period - ensure we use 1st of months
config["train_test_end_date"] = (config["oot_start_date"] - timedelta(days=1)).replace(day=1)
config["train_test_start_date"] = (config["train_test_end_date"] - relativedelta(months=train_test_period_months - 1)).replace(day=1)

# Convert all dates to string format for Spark SQL operations
config["model_train_date_str"] = config["model_train_date"].strftime("%Y-%m-%d")
config["oot_end_date_str"] = config["oot_end_date"].strftime("%Y-%m-%d")
config["oot_start_date_str"] = config["oot_start_date"].strftime("%Y-%m-%d")
config["train_test_end_date_str"] = config["train_test_end_date"].strftime("%Y-%m-%d")
config["train_test_start_date_str"] = config["train_test_start_date"].strftime("%Y-%m-%d")

print("=== MONTHLY DATA CONFIG ===")
pprint.pprint(config)

# Validate the dates make sense for monthly data
print("\n=== DATE VALIDATION ===")
print(f"OOT Period: {config['oot_start_date_str']} to {config['oot_end_date_str']} ({oot_period_months} months)")
print(f"Train-Test Period: {config['train_test_start_date_str']} to {config['train_test_end_date_str']} ({train_test_period_months} months)")


# ## get label store

# In[38]:


# connect to label store
folder_path = "/app/datamart/gold/label_store/"
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
label_store_sdf = spark.read.option("header", "true").parquet(*files_list)

# Ensure snapshot_date is in proper DateType
label_store_sdf = label_store_sdf.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

print("row_count:", label_store_sdf.count())
label_store_sdf.show()


# In[39]:


# extract label store using string dates for Spark SQL
labels_sdf = label_store_sdf.filter(
    (col("snapshot_date") >= to_date(lit(config["train_test_start_date_str"]))) & 
    (col("snapshot_date") <= to_date(lit(config["oot_end_date_str"])))
)

print("extracted labels_sdf", labels_sdf.count(), config["train_test_start_date_str"], config["oot_end_date_str"])


# ## get features

# In[40]:


# connect to feature store
folder_path = "/app/datamart/gold/feature_store/"
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
feature_sdf = spark.read.option("header", "true").parquet(*files_list)

# Ensure snapshot_date is in proper DateType
feature_sdf = feature_sdf.withColumn("snapshot_date", to_date(col("snapshot_date"), "yyyy-MM-dd"))

print("row_count:", feature_sdf.count())
feature_sdf.show()


# In[41]:


# extract feature store using string dates
feature_sdf = feature_sdf.filter(
    (col("snapshot_date") >= to_date(lit(config["train_test_start_date_str"]))) & 
    (col("snapshot_date") <= to_date(lit(config["oot_end_date_str"])))
)

print("extracted feature_sdf", feature_sdf.count(), config["train_test_start_date_str"], config["oot_end_date_str"])
feature_sdf.show()


# ## prepare data for modeling

# In[43]:


# prepare data for modeling
data_sdf = labels_sdf.join(feature_sdf, on=["Customer_ID", "snapshot_date"], how="inner")
data_pdf = data_sdf.toPandas()
data_pdf


# In[60]:


# split data into train - test - oot using Spark DataFrame
oot_sdf = data_sdf.filter(
    (col("snapshot_date") >= to_date(lit(config["oot_start_date_str"]))) & 
    (col("snapshot_date") <= to_date(lit(config["oot_end_date_str"])))
)

train_test_sdf = data_sdf.filter(
    (col("snapshot_date") >= to_date(lit(config["train_test_start_date_str"]))) & 
    (col("snapshot_date") <= to_date(lit(config["train_test_end_date_str"])))
)

# Convert to pandas for modeling
oot_pdf = oot_sdf.toPandas()
train_test_pdf = train_test_sdf.toPandas()

# Define what columns are NOT features
non_feature_cols = [
    'Customer_ID', 'snapshot_date', 'label', 'loan_id', 
    'label_definition', 'label_snapshot_date', 'feature_snapshot_date', 'Occupation',
    'mob', 'dpd', 'dpd_mean', 'dpd_max', 'Loan_overdue_amt_sum', 'Loan_overdue_amt_mean',
    'Loan_overdue_amt_max', 'Loan_amt_sum', 'Loan_amt_mean', 'Loan_amt_std',
    'Loan_balance_sum', 'Loan_balance_mean', 'loan_count', 'Delay_from_due_date',
    'clickstream_total_events',
    'Age'
]

# All other columns are features 
feature_cols = [column for column in train_test_pdf.columns if column not in non_feature_cols]

X_oot = oot_pdf[feature_cols]
y_oot = oot_pdf["label"]
X_train, X_test, y_train, y_test = train_test_split(
    train_test_pdf[feature_cols], train_test_pdf["label"], 
    test_size= 1 - config["train_test_ratio"],
    random_state=88,
    shuffle=True,
    stratify=train_test_pdf["label"]
)

print('X_train', X_train.shape[0])
print('X_test', X_test.shape[0])
print('X_oot', X_oot.shape[0])
print('y_train', y_train.shape[0], round(y_train.mean(),2))
print('y_test', y_test.shape[0], round(y_test.mean(),2))
print('y_oot', y_oot.shape[0], round(y_oot.mean(),2))

X_train.head()


# In[62]:


oot_sdf.show()


# In[59]:


# Check feature columns and first few rows
print("Feature columns in X_train:")
print(X_train.columns.tolist())
print(f"\nNumber of features: {len(X_train.columns)}")

# Display the first few rows with all columns
print("\nX_train head (all columns):")
X_train.head()


# In[63]:


train_test_sdf.show()


# ## preprocess data

# In[85]:


scaler = StandardScaler() 
X_train_processed = scaler.fit_transform(X_train.fillna(0))
X_test_processed = scaler.transform(X_test.fillna(0))
X_oot_processed = scaler.transform(X_oot.fillna(0))

print('X_train_processed', X_train_processed.shape[0])
print('X_test_processed', X_test_processed.shape[0])
print('X_oot_processed', X_oot_processed.shape[0])

pd.DataFrame(X_train_processed)


# ## train model

# In[86]:


# Define the XGBoost classifier
xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)

# Define the hyperparameter space to search
param_dist = {
    'n_estimators': [25, 50],
    'max_depth': [2, 3],  # lower max_depth to simplify the model
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Create a scorer based on AUC score
auc_scorer = make_scorer(roc_auc_score)

# Set up the random search with cross-validation
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    scoring=auc_scorer,
    n_iter=100,  # Number of iterations for random search
    cv=3,       # Number of folds in cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1   # Use all available cores
)

# Perform the random search
random_search.fit(X_train_processed, y_train)

# Output the best parameters and best score
print("Best parameters found: ", random_search.best_params_)
print("Best AUC score: ", random_search.best_score_)

# Evaluate the model on the train set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_train_processed)[:, 1]
train_auc_score = roc_auc_score(y_train, y_pred_proba)
print("Train AUC score: ", train_auc_score)

# Evaluate the model on the test set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test_processed)[:, 1]
test_auc_score = roc_auc_score(y_test, y_pred_proba)
print("Test AUC score: ", test_auc_score)

# Evaluate the model on the oot set
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_oot_processed)[:, 1]
oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
print("OOT AUC score: ", oot_auc_score)

print("TRAIN GINI score: ", round(2*train_auc_score-1,3))
print("Test GINI score: ", round(2*test_auc_score-1,3))
print("OOT GINI score: ", round(2*oot_auc_score-1,3))


# ## prepare model artefact to save

# In[95]:


model_artefact = {}

model_artefact['model'] = best_model
model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
model_artefact['preprocessing_transformers'] = {}
model_artefact['preprocessing_transformers']['stdscaler'] = scaler
model_artefact["feature_names"] = X_train.columns.tolist()
model_artefact['data_dates'] = config
model_artefact['data_stats'] = {}
model_artefact['data_stats']['X_train'] = X_train.shape[0]
model_artefact['data_stats']['X_test'] = X_test.shape[0]
model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
model_artefact['data_stats']['y_test'] = round(y_test.mean(),2)
model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
model_artefact['results'] = {}
model_artefact['results']['auc_train'] = train_auc_score
model_artefact['results']['auc_test'] = test_auc_score
model_artefact['results']['auc_oot'] = oot_auc_score
model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
model_artefact['results']['gini_test'] = round(2*test_auc_score-1,3)
model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
model_artefact['hp_params'] = random_search.best_params_


pprint.pprint(model_artefact)


# ## save artefact to model bank

# In[96]:


# create model_bank dir
model_bank_directory = "/app/model_bank/"

if not os.path.exists(model_bank_directory):
    os.makedirs(model_bank_directory)


# In[97]:


# Full path to the file
file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

# Write the model to a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(model_artefact, file)

print(f"Model saved to {file_path}")


# ## test load pickle and make model inference

# In[98]:


# Load the model from the pickle file
with open(file_path, 'rb') as file:
    loaded_model_artefact = pickle.load(file)

y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_processed)[:, 1]
oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
print("OOT AUC score: ", oot_auc_score)

print("Model loaded successfully!")


# In[ ]:




