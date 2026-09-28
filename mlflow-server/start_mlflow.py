import os
import subprocess
import sys

from dotenv import load_dotenv


load_dotenv()

required_variables = [
    "DATABASE_URL",
    "MLFLOW_ARTIFACT_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_S3_ENDPOINT_URL",
]

missing = [
    variable
    for variable in required_variables
    if not os.getenv(variable)
]

if missing:
    raise RuntimeError(
        "Missing environment variables: "
        + ", ".join(missing)
    )


database_url = os.environ["DATABASE_URL"]
artifact_bucket = os.environ["MLFLOW_ARTIFACT_BUCKET"]

command = [
    sys.executable,
    "-m",
    "mlflow",
    "server",

    "--backend-store-uri",
    database_url,

    "--artifacts-destination",
    f"s3://{artifact_bucket}",

    "--host",
    "127.0.0.1",

    "--port",
    "5000",

    "--workers",
    "1",
]

print("Starting shared MLflow server...")
print("Open: http://127.0.0.1:5000")

subprocess.run(command, check=True)