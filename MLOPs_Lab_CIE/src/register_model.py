import os
import json
import mlflow

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

def register_model():
    """
    Registers the model from the latest run into the MLflow Model Registry.
    """
    print("Starting model registration...")
    
    # Path to the run ID file
    run_id_path = os.path.join("MLOPs_Lab_CIE", "results", "step1_s1.json")
    
    if not os.path.exists(run_id_path):
        print(f"Error: Run ID file not found at '{run_id_path}'. Please run train.py first.")
        return

    with open(run_id_path, 'r') as f:
        run_id = json.load(f)['run_id']
    
    model_name = "cie-lab-model"
    model_uri = f"runs:/{run_id}/model"
    
    print(f"Registering model from run ID '{run_id}' to model name '{model_name}'...")
    
    # Register the model
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    
    print(f"Model registered successfully as version {model_version.version}.")
    
    # Save the model version for the next step
    output_path = os.path.join("MLOPs_Lab_CIE", "results", "step2_s3.json")
    with open(output_path, 'w') as f:
        json.dump({"model_version": model_version.version}, f)
        
    print(f"Model version '{model_version.version}' saved to '{output_path}'.")

if __name__ == "__main__":
    register_model()