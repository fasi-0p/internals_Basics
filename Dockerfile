# 1. Base Image: Use a slim Python image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the project files into the container
# We copy the requirements file first to leverage Docker's layer caching
COPY MLOPs_Lab_CIE/requirements.txt .
COPY MLOPs_Lab_CIE ./MLOPs_Lab_CIE

# 4. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Set the default command to run when the container starts
# This will run the full pipeline: train, register, promote, and predict.
CMD ["sh", "-c", "python MLOPs_Lab_CIE/src/train.py && python MLOPs_Lab_CIE/src/register_model.py && python MLOPs_Lab_CIE/src/promote_model.py && python MLOPs_Lab_CIE/src/predict_cli.py"]