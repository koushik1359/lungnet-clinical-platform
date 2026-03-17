import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments.")
for exp in experiments:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f"Experiment: {exp.name} (ID: {exp.experiment_id}) - Runs: {len(runs)}")
    if not runs.empty:
        print(runs[['run_id', 'status', 'start_time', 'end_time']])
