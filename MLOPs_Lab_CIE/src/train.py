import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

# Set MLflow tracking URI to a local directory
# This will create a folder 'mlruns' in your project directory
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("cie-mlops-lab")

def train_model():
    """
    Trains a RandomForestClassifier, logs the experiment with MLflow,
    and saves the run ID.
    """
    print("Starting the training process...")
    
    # Load dataset
    data_path = os.path.join("MLOPs_Lab_CIE", "data", "training_data.csv")
    df = pd.read_csv(data_path)
    
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start an MLflow run
    with mlflow.start_run() as run:
        print("MLflow run started.")
        
        # Model parameters
        n_estimators = 100
        max_depth = 10
        random_state = 42
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Train the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Accuracy: {accuracy:.4f}")
        print("Model and metrics logged to MLflow.")
        
        # Save the run ID for subsequent steps
        run_id = run.info.run_id
        output_path = os.path.join("MLOPs_Lab_CIE", "results", "step1_s1.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({"run_id": run_id}, f)
            
        print(f"Run ID '{run_id}' saved to '{output_path}'")

if __name__ == "__main__":
    train_model()