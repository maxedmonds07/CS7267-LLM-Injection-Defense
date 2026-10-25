"""Start the shared MLflow server with an optional live Neon connection URL."""

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request

from dotenv import load_dotenv


def get_database_url():
    """Fetch the current Neon URL via REST, or use static DATABASE_URL."""
    project_id = os.getenv("NEON_PROJECT_ID", "").strip()
    if not project_id:
        return os.getenv("DATABASE_URL", "").strip()

    api_key = os.getenv("NEON_API_KEY", "").strip()
    database_name = os.getenv("NEON_DATABASE_NAME", "").strip()
    role_name = os.getenv("NEON_ROLE_NAME", "").strip()
    missing = [
        name
        for name, value in (
            ("NEON_API_KEY", api_key),
            ("NEON_DATABASE_NAME", database_name),
            ("NEON_ROLE_NAME", role_name),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("Missing Neon environment variables: " + ", ".join(missing))

    params = {"database_name": database_name, "role_name": role_name}
    branch_id = os.getenv("NEON_BRANCH_ID", "").strip()
    if branch_id:
        params["branch_id"] = branch_id

    url = (
        "https://console.neon.tech/api/v2/projects/"
        + quote(project_id, safe="")
        + "/connection_uri?"
        + urlencode(params)
    )
    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )

    try:
        with urllib_request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"Neon API returned HTTP {exc.code}. Check your API key, project/branch, "
            "and that a compute endpoint exists."
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("Unable to reach the Neon management API.") from exc

    database_url = payload.get("uri", "").strip()
    if not database_url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("Neon API returned an invalid PostgreSQL connection URL.")
    # SQLAlchemy defaults postgresql:// to psycopg2; this project installs psycopg v3.
    return "postgresql+psycopg://" + database_url.split("://", 1)[1]


def main():
    load_dotenv(Path(__file__).with_name(".env"))

    database_url = get_database_url()
    required_variables = [
        "MLFLOW_ARTIFACT_BUCKET",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_S3_ENDPOINT_URL",
    ]
    missing = [variable for variable in required_variables if not os.getenv(variable)]
    if not database_url:
        missing.insert(0, "DATABASE_URL or NEON_PROJECT_ID")
    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))

    # Leave the connection string in process memory; no credentials are written to .env.
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        database_url,
        "--artifacts-destination",
        f"s3://{os.environ['MLFLOW_ARTIFACT_BUCKET']}",
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


if __name__ == "__main__":
    main()
