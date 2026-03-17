import mlflow
import os
import shutil

def check_location(uri, label):
    print(f"\n--- Checking {label} ({uri}) ---")
    try:
        mlflow.set_tracking_uri(uri)
        exps = mlflow.search_experiments()
        print(f"Found {len(exps)} experiments.")
        for e in exps:
            runs = mlflow.search_runs(experiment_ids=[e.experiment_id])
            print(f"  - Exp: '{e.name}' (ID: {e.experiment_id}) | Runs: {len(runs)}")
            if not runs.empty:
                print(f"    Latest Run: {runs.iloc[0]['run_id']} | Status: {runs.iloc[0]['status']}")
    except Exception as err:
        print(f"  - Error checking {label}: {err}")

# Check local mlruns folder using absolute path (Correct for Windows)
mlruns_path = os.path.abspath("mlruns")
if os.path.exists(mlruns_path):
    check_location(mlruns_path, "Local mlruns folder")
else:
    print(f"\n[!] 'mlruns' folder NOT found in {os.getcwd()}")

# Check SQLite DB
db_path = os.path.abspath("mlflow.db")
if os.path.exists(db_path):
    check_location(f"sqlite:///{db_path}", "SQLite Database")
else:
    print(f"\n[!] 'mlflow.db' NOT found in {os.getcwd()}")
