import argparse
from mlflow.tracking import MlflowClient

def main(experiment_name: str):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"❌ Experiment not found: {experiment_name}")
        return

    # Find the single most‐recent run
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"❌ No runs found for experiment: {experiment_name}")
        return

    run = runs[0]
    print(f"\n🏷  Experiment: {experiment_name!r} (ID {exp.experiment_id})")
    print(f"🔖 Latest run ID: {run.info.run_id}")
    print(f"⏰ Start time:   {run.info.start_time}\n")

    # Params
    print("Parameters:")
    for k, v in run.data.params.items():
        print(f"  • {k} = {v}")
    # Metrics
    print("\nMetrics:")
    for k, v in run.data.metrics.items():
        print(f"  • {k} = {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show latest MLflow run metrics for a given experiment"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="MLflow experiment name (e.g. Experiment_4)",
    )
    args = parser.parse_args()
    main(args.experiment)
