import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Custom MAPE function to avoid division by zero
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_models():
    """
    Trains RandomForest and GradientBoosting regressors, logs experiments to MLflow,
    compares them by RMSE, and saves the results as per the PDF specification.
    """
    print("Starting Task 1: Experiment Tracking & Model Comparison")

    # MLflow configuration
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "swimsync-lap-time-seconds"
    mlflow.set_experiment(experiment_name)

    # Load data
    data_path = os.path.join("MLOPs_Lab_CIE", "data", "training_data.csv")
    df = pd.read_csv(data_path)
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_to_train = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    results_list = []
    best_model_info = {"name": None, "rmse": float('inf'), "run_id": None}

    for name, model in models_to_train.items():
        with mlflow.start_run(run_name=f"{name}_training") as run:
            print(f"Training {name} model...")

            # Set tag
            mlflow.set_tag("team", "ml_engineering")

            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # Log params and metrics
            mlflow.log_params(model.get_params())
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            
            # Log the model
            mlflow.sklearn.log_model(model, name)
            
            print(f"  {name} Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

            model_metrics = {
                "name": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "mape": mape
            }
            results_list.append(model_metrics)

            # Check if this is the best model
            if rmse < best_model_info["rmse"]:
                best_model_info["name"] = name
                best_model_info["rmse"] = rmse
                best_model_info["run_id"] = run.info.run_id

    # Prepare final JSON output for step1_s1.json
    output = {
        "experiment_name": experiment_name,
        "models": results_list,
        "best_model": best_model_info["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best_model_info["rmse"]
    }
    
    # Save the run ID of the best model for the next step
    # Although not explicitly in step1_s1.json, we need it for Task 3
    # Let's create a separate file for it to keep things clean and traceable
    best_run_output = {
        "best_model_run_id": best_model_info["run_id"],
        "best_model_name": best_model_info["name"]
    }
    with open(os.path.join("MLOPs_Lab_CIE", "results", "best_model_run_id.json"), 'w') as f:
        json.dump(best_run_output, f, indent=4)

    # Save Task 1 results
    output_path = os.path.join("MLOPs_Lab_CIE", "results", "step1_s1.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\nBest model is '{best_model_info['name']}' with RMSE: {best_model_info['rmse']:.4f}")
    print(f"Task 1 results saved to '{output_path}'")

if __name__ == "__main__":
    train_models()