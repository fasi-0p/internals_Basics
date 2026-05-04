import os
import json
import mlflow

def register_best_model():
    """
    Registers the best model from Task 1 into the MLflow Model Registry
    and saves the results as per the PDF specification for Task 3.
    """
    print("Starting Task 3: Model Versioning")
    
    # MLflow configuration
    mlflow.set_tracking_uri("file:./mlruns")

    # --- Load info about the best model ---
    run_id_path = os.path.join("MLOPs_Lab_CIE", "results", "best_model_run_id.json")
    with open(run_id_path, 'r') as f:
        best_model_info = json.load(f)
    
    run_id = best_model_info["best_model_run_id"]
    model_name_from_run = best_model_info["best_model_name"]
    
    model_uri = f"runs:/{run_id}/{model_name_from_run}"
    registered_model_name = "swimsync-lap-time-seconds-predictor"

    print(f"Registering model '{model_name_from_run}' from run '{run_id}' as '{registered_model_name}'")
    
    # Register the model
    model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    
    print(f"Model registered as Version: {model_version.version}")

    # --- Save results for Task 3 ---
    task1_results_path = os.path.join("MLOPs_Lab_CIE", "results", "step1_s1.json")
    with open(task1_results_path, 'r') as f:
        task1_results = json.load(f)
        
    output = {
        "registered_model_name": registered_model_name,
        "version": model_version.version,
        "run_id": run_id,
        "source_metric": "rmse",
        "source_metric_value": task1_results["best_metric_value"]
    }
    
    output_path = os.path.join("MLOPs_Lab_CIE", "results", "step3_s6.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"Task 3 results saved to '{output_path}'")

if __name__ == "__main__":
    register_best_model()