import os
import json
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

def promote_model():
    """
    Promotes the latest registered model version to the 'Staging' stage.
    """
    print("Starting model promotion...")

    # Path to the model version file
    version_path = os.path.join("MLOPs_Lab_CIE", "results", "step2_s3.json")
    
    if not os.path.exists(version_path):
        print(f"Error: Model version file not found at '{version_path}'. Please run register_model.py first.")
        return
        
    with open(version_path, 'r') as f:
        model_version = json.load(f)['model_version']

    client = MlflowClient()
    model_name = "cie-lab-model"
    
    print(f"Promoting model '{model_name}' version '{model_version}' to 'Staging'...")
    
    # Transition the model to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging"
    )
    
    print("Model promoted to Staging successfully.")
    
    # Save a confirmation for the next step
    output_path = os.path.join("MLOPs_Lab_CIE", "results", "step3_s6.json")
    with open(output_path, 'w') as f:
        json.dump({"promoted_to_staging": True, "model_version": model_version}, f)

    print(f"Promotion confirmation saved to '{output_path}'.")

if __name__ == "__main__":
    promote_model()