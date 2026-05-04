import os
import argparse
import pandas as pd
import json
import mlflow

def predict():
    """
    Loads the best model from a specific run and makes a prediction
    based on command-line arguments.
    """
    print("Starting Task 2: Prediction CLI")
    
    # MLflow configuration
    mlflow.set_tracking_uri("file:./mlruns")

    # --- Load the best model from the previous step ---
    run_id_path = os.path.join("MLOPs_Lab_CIE", "results", "best_model_run_id.json")
    with open(run_id_path, 'r') as f:
        best_model_info = json.load(f)
    
    run_id = best_model_info["best_model_run_id"]
    model_name = best_model_info["best_model_name"]
    
    # Load the model artifact from the specific run
    try:
        model_uri = f"runs:/{run_id}/{model_name}"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model '{model_name}' from run ID '{run_id}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Argument parsing for prediction input ---
    parser = argparse.ArgumentParser(description="SwimSync Lap Time Predictor")
    parser.add_argument("--stroke_rate", type=float, required=True)
    parser.add_argument("--drag_coefficient", type=float, required=True)
    parser.add_argument("--turn_time_ms", type=float, required=True)
    parser.add_argument("--pool_length_m", type=float, required=True)
    args = parser.parse_args()

    # --- Create a DataFrame for prediction ---
    input_data = {
        "stroke_rate": [args.stroke_rate],
        "drag_coefficient": [args.drag_coefficient],
        "turn_time_ms": [args.turn_time_ms],
        "pool_length_m": [args.pool_length_m]
    }
    input_df = pd.DataFrame(input_data)

    # --- Make prediction ---
    prediction = model.predict(input_df)[0]
    print(f"Prediction for input data: {prediction}")

    # --- Save results for Task 2 ---
    test_input = vars(args) # Convert argparse namespace to dict
    output = {
        "image_name": "swimsync-predictor",
        "image_tag": "v1",
        "base_image": "python:3.12-slim",
        "test_input": test_input,
        "prediction": prediction
    }
    
    output_path = os.path.join("MLOPs_Lab_CIE", "results", "step2_s3.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"Task 2 results saved to '{output_path}'")

if __name__ == "__main__":
    predict()