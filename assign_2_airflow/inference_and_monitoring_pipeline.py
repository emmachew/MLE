import subprocess
import sys

print("\n🚀 RUNNING INFERENCE...\n")
inference_cmd = [
    "python", "/app/utils/model_inference.py"
]

res1 = subprocess.run(inference_cmd)
if res1.returncode != 0:
    print("❌ Inference failed. Exiting pipeline.")
    sys.exit(res1.returncode)

print("\n✅ Inference completed.\n")
print("\n🚀 RUNNING MODEL MONITORING...\n")

monitor_cmd = [
    "python", "/app/utils/monitor_predictions.py",
    "--model_label", "credit_model_2024_12_01.pkl",
    "--snapshotdate", "2024_12_01",
    "--baseline_date", "2024_06_01",
    "--pred_path", "/app/datamart/gold/model_predictions/credit_model_2024_12_01/credit_model_2024_12_01.parquet",
    "--baseline_path", "/app/datamart/gold/model_predictions/credit_model_2024_06_01/credit_model_2024_06_01.parquet",
    "--output_dir", "/app/datamart/gold/model_monitoring/credit_model_2024_12_01",
    "--labels_path", "/app/datamart/gold/label_store"
]

res2 = subprocess.run(monitor_cmd)
if res2.returncode != 0:
    print("❌ Monitoring failed.")
    sys.exit(res2.returncode)

print("\n✅ Monitoring completed. Pipeline finished successfully! 🎉")
