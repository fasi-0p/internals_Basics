import os
import pandas as pd
import mlflow
import click
import json

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")

@click.command()
@click.option("--data-path", default=os.path.join("MLOPs_Lab_CIE", "data", "new_data.csv"), help="Path to the new data for prediction.")
def predict(data_path):
    """
    Loads the 'Staging' model from MLflow and makes predictions on new data.
    """
    print("Starting prediction process...")
    model_name = "cie-lab-model"
    stage = 'Staging'

    try:
        # Load the model from the Staging environment
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{stage}"
        )
        print(f"Model '{model_name}' (Stage: {stage}) loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure a model has been trained, registered, and promoted to Staging.")
        return
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at '{data_path}'.")
        return
        
    # Load new data
    new_data = pd.read_csv(data_path)
    
    # Make predictions
    predictions = model.predict(new_data)
    
    print("\nPredictions:")
    print(predictions)
    
    # Save predictions to a file
    output_path = os.path.join("MLOPs_Lab_CIE", "results", "step4_s7.json")
    with open(output_path, 'w') as f:
        json.dump(predictions.tolist(), f)
        
    print(f"\nPredictions saved to '{output_path}'.")

if __name__ == "__main__":
    predict()
