import mlflow
import os

paths = [
    r"C:\Users\koush\OneDrive\Desktop\ML Projects\med_ai_platform\mlflow.db",
    r"C:\Users\koush\OneDrive\Desktop\ML%20Projects\med_ai_platform\mlflow.db"
]

for p in paths:
    if os.path.exists(p):
        uri = f"sqlite:///{p}"
        try:
            mlflow.set_tracking_uri(uri)
            exps = mlflow.search_experiments()
            for e in exps:
                runs = mlflow.search_runs(experiment_ids=[e.experiment_id])
                print(f"PATH: {p} | Experiment: {e.name} | RUNS: {len(runs)}")
        except Exception as e:
            print(f"PATH: {p} | ERROR: {e}")
    else:
        print(f"PATH: {p} | DOES NOT EXIST")
