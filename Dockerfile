# 1. Base Image as specified in the PDF
FROM python:3.12-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy project files
COPY MLOPs_Lab_CIE/requirements.txt .
COPY MLOPs_Lab_CIE ./MLOPs_Lab_CIE

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Set the entrypoint for the CLI tool
# This allows passing arguments directly to the script via `docker run`
ENTRYPOINT ["python", "MLOPs_Lab_CIE/src/predict_cli.py"]