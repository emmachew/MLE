#!/usr/bin/env python3
# coding: utf-8
"""
Train Random Forest model for loan default prediction
Includes embedded preprocessing code.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import joblib
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import randint, uniform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def preprocess_features_for_lr(X_train, X_test_list, categorical_cols=None):
    """
    Preprocess features for tree-based models (Random Forest, XGBoost)
    Returns: X_train_processed, [X_test_processed_list], encoder, feature_names
    """
    if categorical_cols is None:
        categorical_cols = []
    
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    # For tree models, we only need to encode categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)
        ],
        remainder='passthrough'  # Keep numerical columns as is
    )
    
    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed_list = []
    for X_test in X_test_list:
        X_test_processed = preprocessor.transform(X_test)
        X_test_processed_list.append(X_test_processed)
    
    # Get feature names
    feature_names = numerical_cols.copy()
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    
    return X_train_processed, X_test_processed_list, preprocessor, feature_names


def preprocess_features_for_lr(X_train, X_test_list, categorical_cols=None):
    """
    Preprocess features for Logistic Regression
    Returns: X_train_processed, [X_test_processed_list], scaler, feature_names
    """
    if categorical_cols is None:
        categorical_cols = []
    
    numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse=False), categorical_cols)
        ]
    )
    
    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_processed_list = []
    for X_test in X_test_list:
        X_test_processed = preprocessor.transform(X_test)
        X_test_processed_list.append(X_test_processed)
    
    # Get feature names
    feature_names = numerical_cols.copy()
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_features)
    
    return X_train_processed, X_test_processed_list, preprocessor, feature_names


def build_config(train_date_str, train_months=8, val_months=2, test_months=2, oot_months=2):
    model_train_date = datetime.strptime(train_date_str, "%Y-%m-%d")
    cfg = {}
    cfg["model_train_date"] = model_train_date
    cfg["oot_end"] = model_train_date - timedelta(days=1)
    cfg["oot_start"] = model_train_date - relativedelta(months=oot_months)
    cfg["test_end"] = cfg["oot_start"] - timedelta(days=1)
    cfg["test_start"] = cfg["oot_start"] - relativedelta(months=test_months)
    cfg["val_end"] = cfg["test_start"] - timedelta(days=1)
    cfg["val_start"] = cfg["test_start"] - relativedelta(months=val_months)
    cfg["train_end"] = cfg["val_start"] - timedelta(days=1)
    cfg["train_start"] = cfg["val_start"] - relativedelta(months=train_months)
    cfg["data_start_date"] = cfg["train_start"]
    cfg["data_end_date"]   = cfg["oot_end"]
    return cfg

def start_spark(app_name="rf_training", master="local[*]"):   
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def stratified_sample_spark(sdf, label_col, frac, seed=42):
    values = [r[label_col] for r in sdf.select(label_col).distinct().collect()]
    fractions = {int(v): float(frac) for v in values}
    logger.info(f"Sampling fractions per label: {fractions}")
    sampled = sdf.sampleBy(label_col, fractions, seed)
    logger.info(f"Sampled rows: {sampled.count()}")
    return sampled

def sdf_to_pandas(sdf):
    pdf = sdf.toPandas()
    logger.info(f"Converted Spark DF to pandas: {pdf.shape}")
    return pdf

def evaluate_model(model, X, y, threshold=0.5):
    """Evaluate model with all required metrics"""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, proba)
    }
    
    # Add confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    metrics.update({
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    })
    
    return metrics

def main(args):
    cfg = build_config(args.train_date, train_months=args.train_months,
                       val_months=args.val_months, test_months=args.test_months, oot_months=args.oot_months)
    logger.info(f"Configuration: Train={cfg['train_start'].date()}→{cfg['train_end'].date()}")
    
    spark = start_spark(app_name="loan_rf_train")

    # Read data
    features_sdf = spark.read.parquet(args.features_path).filter(
        (col("snapshot_date") >= F.lit(cfg["data_start_date"].date())) &
        (col("snapshot_date") <= F.lit(cfg["data_end_date"].date()))
    )
    labels_sdf = spark.read.parquet(args.labels_path).filter(
        (col("snapshot_date") >= F.lit(cfg["data_start_date"].date())) &
        (col("snapshot_date") <= F.lit(cfg["data_end_date"].date()))
    )

    joined = features_sdf.join(labels_sdf, on=["customer_id", "snapshot_date"], how="inner")

    # Determine label column
    label_col = args.label_col
    if label_col not in joined.columns:
        for potential_col in ["label", "default", "is_default"]:
            if potential_col in joined.columns:
                label_col = potential_col
                break
        if label_col not in joined.columns:
            raise ValueError(f"Label column '{args.label_col}' not found.")

    # Convert DPD (Days Past Due) to binary labels if needed
    # Check if labels are DPD values (0, 30) instead of binary (0, 1)
    distinct_labels = [row[label_col] for row in joined.select(label_col).distinct().collect()]
    if set(distinct_labels) == {0, 30}:
        logger.info("Converting DPD labels (0, 30) to binary labels (0, 1)")
        joined = joined.withColumn(label_col, F.when(F.col(label_col) > 0, 1).otherwise(0))
    elif 30 in distinct_labels:
        logger.warning(f"Label column contains value 30 - may need DPD to binary conversion")

    # Create time-based splits
    splits = {}
    splits["train"] = joined.filter((col("snapshot_date") >= F.lit(cfg["train_start"].date())) &
                                   (col("snapshot_date") <= F.lit(cfg["train_end"].date())))
    splits["val"] = joined.filter((col("snapshot_date") >= F.lit(cfg["val_start"].date())) &
                                 (col("snapshot_date") <= F.lit(cfg["val_end"].date())))
    splits["test"] = joined.filter((col("snapshot_date") >= F.lit(cfg["test_start"].date())) &
                                  (col("snapshot_date") <= F.lit(cfg["test_end"].date())))
    splits["oot"] = joined.filter((col("snapshot_date") >= F.lit(cfg["oot_start"].date())) &
                                 (col("snapshot_date") <= F.lit(cfg["oot_end"].date())))

    # Sampling & convert to pandas
    sampled_pdfs = {}
    for k, sdf in splits.items():
        # Check if we have both classes before stratified sampling
        label_counts = sdf.groupBy(label_col).count().collect()
        class_counts = {row[label_col]: row['count'] for row in label_counts}
        
        logger.info(f"Split '{k}' class distribution: {class_counts}")
        
        if args.sample_frac < 1.0 and len(class_counts) > 1:
            # We have multiple classes - use stratified sampling
            sdf_cast = sdf.withColumn(label_col, col(label_col).cast("int"))
            sampled = stratified_sample_spark(sdf_cast, label_col, args.sample_frac, seed=args.random_state)
        else:
            # Single class or no sampling - use simple sampling
            if len(class_counts) <= 1:
                logger.warning(f"Split '{k}' has only one class: {class_counts}")
            sampled = sdf.sample(args.sample_frac, seed=args.random_state) if args.sample_frac < 1.0 else sdf
        
        pdf = sdf_to_pandas(sampled)
        sampled_pdfs[k] = pdf

    spark.stop()
    logger.info("Stopped Spark")

    # Feature columns from your importance list
    feature_cols = [
        'Age', 'Occupation', 'Delay_from_due_date', 'Outstanding_Debt', 
        'Amount_invested_monthly', 'Interest_Rate', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Loan_Type_Home_Loan', 'Loan_Type_Personal_Loan',
        'Loan_Type_Student_Loan', 'Loan_Type_Auto_Loan', 'Loan_Type_Business_Loan',
        'Loan_Type_Credit-Builder_Loan', 'Loan_Type_Home_Equity_Loan',
        'Loan_Type_Debt_Consolidation_Loan', 'Loan_Type_Mortgage_Loan',
        'Loan_Type_Not_Specified', 'Loan_Type_Payday_Loan', 'Loan_amt_sum',
        'Loan_amt_mean', 'Loan_amt_std', 'Loan_tenure_mean', 'Loan_tenure_max',
        'Loan_overdue_amt_sum', 'Loan_overdue_amt_mean', 'Loan_overdue_amt_max',
        'Loan_balance_sum', 'Loan_balance_mean', 'dpd_mean', 'dpd_max', 'loan_count',
        'clickstream_total_events', 'clickstream_fe_5_mean', 'clickstream_fe_5_sum',
        'clickstream_fe_5_std', 'clickstream_fe_9_mean', 'clickstream_fe_9_min',
        'clickstream_fe_4_mean', 'clickstream_fe_4_min', 'clickstream_fe_10_mean',
        'clickstream_fe_10_min'
    ]

    # Categorical columns
    categorical_cols = [
        'Occupation', 'Loan_Type_Home_Loan', 'Loan_Type_Personal_Loan',
        'Loan_Type_Student_Loan', 'Loan_Type_Auto_Loan', 'Loan_Type_Business_Loan',
        'Loan_Type_Credit-Builder_Loan', 'Loan_Type_Home_Equity_Loan',
        'Loan_Type_Debt_Consolidation_Loan', 'Loan_Type_Mortgage_Loan',
        'Loan_Type_Not_Specified', 'Loan_Type_Payday_Loan'
    ]

    # Build X/y datasets
    X_train = sampled_pdfs["train"][feature_cols].copy()
    y_train = sampled_pdfs["train"][label_col].astype(int).copy()
    X_val = sampled_pdfs["val"][feature_cols].copy()
    y_val = sampled_pdfs["val"][label_col].astype(int).copy()
    X_test = sampled_pdfs["test"][feature_cols].copy()
    y_test = sampled_pdfs["test"][label_col].astype(int).copy()
    X_oot = sampled_pdfs["oot"][feature_cols].copy()
    y_oot = sampled_pdfs["oot"][label_col].astype(int).copy()

    # Preprocessing for tree models
    X_train_rf, [X_val_rf, X_test_rf, X_oot_rf], encoder, feature_names = preprocess_features_for_lr(
        X_train, [X_val, X_test, X_oot], categorical_cols=categorical_cols
    )

    # Class weights
    cw_vals = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y_train)
    class_weight_dict = {0: cw_vals[0], 1: cw_vals[1]}

    # Hyperparameter space for Random Forest
    rf_param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # Base Random Forest model
    rf_base = RandomForestClassifier(
        class_weight=class_weight_dict,
        random_state=args.random_state,
        n_jobs=-1
    )

    # MLflow setup
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # RandomizedSearchCV with recall scoring
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_dist,
        n_iter=args.n_iter,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        random_state=args.random_state,
        return_train_score=True
    )

    with mlflow.start_run(run_name=f"RandomForest_{args.train_date}"):
        # Log parameters
        mlflow.log_params({
            "model_type": "RandomForest",
            "train_date": args.train_date,
            "sample_frac": args.sample_frac,
            "cv_folds": args.cv_folds,
            "n_iter": args.n_iter,
            "scoring_metric": "recall"
        })
        
        search.fit(X_train_rf, y_train)
        best = search.best_estimator_
        
        mlflow.log_metric("best_cv_recall", float(search.best_score_))
        for k, v in search.best_params_.items():
            mlflow.log_param(f"best_{k}", v)

        # Evaluate on all datasets with all metrics
        for name, X, y in [("train", X_train_rf, y_train), ("val", X_val_rf, y_val),
                           ("test", X_test_rf, y_test), ("oot", X_oot_rf, y_oot)]:
            metrics = evaluate_model(best, X, y)
            logger.info(f"{name} set metrics: {metrics}")
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{name}_{metric_name}", value)

        # Log model
        artifact_path = "random_forest_model"
        mlflow.sklearn.log_model(best, artifact_path)
        
        # Save preprocessing artifacts
        encoder_path = "encoder.pkl"
        joblib.dump(encoder, encoder_path)
        mlflow.log_artifact(encoder_path, artifact_path="preprocessing")
        os.remove(encoder_path)

        feature_names_path = "feature_names.txt"
        with open(feature_names_path, "w") as f:
            f.write("\n".join(feature_names))
        mlflow.log_artifact(feature_names_path, artifact_path="preprocessing")
        os.remove(feature_names_path)

        # Log feature importance
        importance = best.feature_importances_
        feature_importance_dict = dict(zip(feature_names, importance))
        for feature, imp in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
            mlflow.log_metric(f"feature_importance_{feature}", imp)

    logger.info("Random Forest training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_date", type=str, required=True, help="Training date in YYYY-MM-DD format")
    parser.add_argument("--features_path", type=str, default="/app/datamart/gold/feature_store/")
    parser.add_argument("--labels_path", type=str, default="/app/datamart/gold/label_store/")
    parser.add_argument("--sample_frac", type=float, default=0.3, help="Fraction of data to sample")
    parser.add_argument("--label_col", type=str, default="default", help="Label column name")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://mlflow:5000")
    parser.add_argument("--mlflow_experiment", type=str, default="loan-default-prediction")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of random search iterations")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--train_months", type=int, default=8)
    parser.add_argument("--val_months", type=int, default=2)
    parser.add_argument("--test_months", type=int, default=2)
    parser.add_argument("--oot_months", type=int, default=2)
    args = parser.parse_args()
    main(args)