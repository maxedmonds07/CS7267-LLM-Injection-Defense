import os
import subprocess

import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Connect to local MLflow server
# ---------------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Creates the experiment if it does not already exist
mlflow.set_experiment("MLflow_Example")


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------


def get_git_branch():
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True
        ).strip()
    except Exception:
        return "unknown"


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "unknown"


team_member = os.getenv(
    "MLFLOW_TEAM_MEMBER",
    "example-user"
)


# ---------------------------------------------------------
# Load small built-in dataset
# ---------------------------------------------------------

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ---------------------------------------------------------
# Train a very small model
# ---------------------------------------------------------

model = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


# ---------------------------------------------------------
# Log experiment to MLflow
# ---------------------------------------------------------

with mlflow.start_run(
    run_name="example_decision_tree"
):

    # Log parameters
    mlflow.log_params({
        "model": "DecisionTreeClassifier",
        "max_depth": 2,
        "test_size": 0.25,
        "random_state": 42
    })

    # Log metric
    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # Log metadata
    mlflow.set_tags({
        "team_member": team_member,
        "git_branch": get_git_branch(),
        "git_commit": get_git_commit(),
        "model_status": "example"
    })

    # -----------------------------------------------------
    # Store the actual trained model
    # -----------------------------------------------------

    mlflow.sklearn.log_model(
    sk_model=model,
    name="example_model",
    skops_trusted_types=[
        "sklearn.tree._tree.Tree"
    ])

print("Example MLflow run completed successfully.")
print(f"Accuracy: {accuracy:.4f}")
print("Model stored as: example_model")
print("Open MLflow at: http://127.0.0.1:5000")