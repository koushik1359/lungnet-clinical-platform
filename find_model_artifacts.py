import mlflow
import os

db_path = r"C:\Users\koush\OneDrive\Desktop\ML Projects\med_ai_platform\mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

try:
    experiments = mlflow.search_experiments()
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        for index, run in runs.iterrows():
            print(f"RUN: {run['tags.mlflow.runName']} | ID: {run['run_id']} | Artifacts: {run['artifact_uri']}")
except Exception as e:
    print(f"Error: {e}")

# Also check for .pth files again but better
print("\n--- Searching for .pth files ---")
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".pth"):
            print(os.path.join(root, file))
