# Shared MLflow Setup

## CS7267 – LLM Injection Defense Project

This project uses **MLflow** to centrally track machine-learning experiments performed by different team members.

We do **not** maintain one continuously running cloud MLflow server.

Instead:

- Each team member runs an MLflow server locally on their own computer.
- All local MLflow servers connect to the **same Neon PostgreSQL database**.
- All local MLflow servers connect to the **same Neon Object Storage bucket**.
- Therefore, everyone can see the same experiments, runs, parameters, metrics, and artifacts.
- GitHub continues to handle source-code version control separately.

---

# 1. Architecture

```text
                         GitHub Repository
                    Source Code / Version Control
                              |
               --------------------------------
               |              |               |
           Member 1        Member 2        Member 3-5
               |              |               |
        Local MLflow     Local MLflow     Local MLflow
       127.0.0.1:5000  127.0.0.1:5000  127.0.0.1:5000
               |              |               |
               ---------------|---------------
                              |
                              v
                  Neon PostgreSQL Database
                       FREE TIER
                ------------------------
                Experiments
                Runs
                Parameters
                Metrics
                Tags
                Model metadata

                Free-plan allowance:
                • 0.5 GB database storage
                • 100 CU-hours/project
                • 10 database branches

                              |
                              v

                  Neon Object Storage
                       FREE TIER
                ------------------------
                Trained models
                Plots
                CSV files
                Reports
                Other MLflow artifacts

                Free-plan allowance:
                • 5 GB object storage/project
```

The PostgreSQL database stores MLflow metadata such as experiments, runs, parameters, metrics, and tags.

The Object Storage bucket stores larger files such as trained models, plots, reports, and other artifacts.

For this project, we should avoid unnecessarily storing large datasets or every intermediate trained model.

---

# 2. First-Time Setup

Each collaborator only needs to complete this section once.

## 2.1 Clone the Repository

```bash
git clone https://github.com/maxedmonds07/CS7267-LLM-Injection-Defense.git
```

Enter the repository:

```bash
cd CS7267-LLM-Injection-Defense
```

Enter the MLflow server folder:

```bash
cd mlflow-server
```

---

## 2.2 Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

---

## 2.3 Install MLflow Requirements

```bash
pip install -r requirements.txt
```

The project uses the same `requirements.txt` so that all collaborators use compatible MLflow versions.

Do not independently upgrade MLflow unless the change is made through the repository's `requirements.txt`.

---

## 2.4 Configure the `.env` File

Each team member needs a local:

```text
mlflow-server/.env
```

Copy:

```text
.env.example
```

and populate it with the credentials supplied by the project owner.

The required settings are:

```text
DATABASE_URL=
MLFLOW_ARTIFACT_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
MLFLOW_S3_ENDPOINT_URL=
MLFLOW_TEAM_MEMBER=
```

Example:

```text
DATABASE_URL=your_neon_database_url
MLFLOW_ARTIFACT_BUCKET=your_bucket_name
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
MLFLOW_S3_ENDPOINT_URL=your_storage_endpoint
MLFLOW_TEAM_MEMBER=YourName
```

The Neon database and storage credentials are shared by the team.

`MLFLOW_TEAM_MEMBER` identifies which collaborator created a particular MLflow run.

For example:

```text
MLFLOW_TEAM_MEMBER=Alice
```

or:

```text
MLFLOW_TEAM_MEMBER=Bob
```

---

## 2.5 Never Commit `.env`

The `.env` file contains credentials and must never be pushed to GitHub.

Make sure `.gitignore` contains:

```gitignore
.env
```

Never put database passwords, storage credentials, or access keys inside:

- Python scripts
- Jupyter notebooks
- README files
- Git commits
- Pull Requests
- GitHub Issues

---

## 2.6 Start the MLflow Server

Every time you want to use MLflow, open a terminal and enter:

```bash
cd mlflow-server
```

Activate the virtual environment.

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Start MLflow:

```bash
python start_mlflow.py
```

Open:

```text
http://127.0.0.1:5000
```

Keep this terminal open while running ML experiments.

To stop the server:

```text
Ctrl + C
```

Stopping the server does **not** delete any experiments.

The experiment information is already stored in Neon.

---

# 3. What Each Component Does

## GitHub

GitHub tracks our **source code**.

Examples include:

- Python scripts
- Jupyter notebooks
- Model implementations
- Preprocessing code
- Configuration
- Documentation
- Requirements

GitHub provides:

- Branches
- Commits
- Pull Requests
- Merge history
- Version control

GitHub answers:

> What version of the code are we using?

---

## MLflow

MLflow tracks our **machine-learning experiments**.

Examples include:

- Model type
- Hyperparameters
- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Training configuration
- Git commit
- Git branch
- Team member
- Plots
- Model files
- Other artifacts

MLflow helps answer questions such as:

> What experiments did we run?

> What parameters were used?

> What metrics did each run produce?

> Who ran the experiment?

> Which Git commit produced the result?

---

## Neon PostgreSQL

Neon PostgreSQL is the shared backend database used by MLflow.

It stores information such as:

```text
Experiments
Runs
Parameters
Metrics
Tags
Run status
Model metadata
Artifact locations
```

Every team member connects their local MLflow server to the **same database**.

That is why all collaborators can see the same experiments.

---

## Neon Object Storage

Neon Object Storage stores files associated with MLflow runs.

Examples include:

```text
Trained models
Confusion matrices
ROC curves
PR curves
SHAP plots
Feature importance plots
Classification reports
CSV files
JSON files
Other experiment artifacts
```

All collaborators use the same Object Storage bucket.

---

# 4. How MLflow Actually Works

An important point:

## MLflow does NOT automatically track GitHub activity

The following commands do not automatically create an MLflow run:

```text
git add
git commit
git push
Pull Request
Merge to main
```

MLflow records information only when the Python program communicates with the MLflow tracking server.

For example:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("baseline_models")

with mlflow.start_run():

    mlflow.log_param(
        "model",
        "RandomForest"
    )

    mlflow.log_param(
        "n_estimators",
        200
    )

    mlflow.log_metric(
        "accuracy",
        0.91
    )
```

When this code runs:

```text
Training Script
       |
       v
Local MLflow Server
127.0.0.1:5000
       |
       v
Neon PostgreSQL
       |
       +---- Experiment
       +---- Run
       +---- Parameters
       +---- Metrics
       +---- Tags
```

If artifacts are logged:

```python
mlflow.log_artifact(
    "confusion_matrix.png"
)
```

the flow is:

```text
Training Script
       |
       v
Local MLflow Server
       |
       v
Neon Object Storage
```

---

# 5. Experiments, Runs, Models, and Model Registry

These terms have different meanings.

```text
Experiment
    |
    └── Run
         |
         ├── Parameters
         ├── Metrics
         ├── Tags
         ├── Artifacts
         |
         └── Logged Model (optional)
                  |
                  v
             Model Registry
             (optional)
```

## Experiment

An **experiment** groups related runs.

Example:

```text
baseline_models
```

It might contain:

```text
Logistic Regression run
Random Forest run
XGBoost run
```

---

## Run

A **run** represents one execution of training/evaluation code.

Example:

```text
random_forest_depth_10
```

It may contain:

```text
n_estimators = 200
max_depth = 10
accuracy = 0.91
f1_macro = 0.89
team_member = Alice
git_branch = feature/random-forest
```

Running the same script again normally creates another run.

MLflow does not overwrite the previous run.

---

## Logged Model

A trained model is stored only when we explicitly tell MLflow to store it.

Training:

```python
model.fit(X_train, y_train)
```

does **not** automatically save the model to MLflow.

The model is stored when we use something such as:

```python
mlflow.sklearn.log_model(
    sk_model=model,
    name="model"
)
```

Therefore:

```text
Train model
      |
      X
Model is NOT automatically stored
```

but:

```text
Train model
      |
mlflow.sklearn.log_model(...)
      |
      v
Model stored in Object Storage
```

---

## Model Registry

The Model Registry is used when we want to formally manage selected models.

Example:

```text
PromptInjectionDetector

Version 1
Version 2
Version 3
```

Not every experiment needs to be registered.

For this project, we can use the registry mainly for selected or final models.

---

# 6. Do I Need to Run MLflow Every Time?

## Yes, when you want MLflow tracking

Before running code containing:

```python
mlflow.start_run()
mlflow.log_param()
mlflow.log_metric()
mlflow.log_artifact()
mlflow.sklearn.log_model()
```

the local MLflow server should be running.

Start it with:

```bash
python start_mlflow.py
```

Your training code communicates with:

```text
http://127.0.0.1:5000
```

---

## You do not need to leave MLflow running continuously

Once an experiment is logged:

```text
Experiment
    |
    v
Neon PostgreSQL / Object Storage
```

the information remains stored.

You can stop the local MLflow server.

Later:

```bash
python start_mlflow.py
```

and all previous experiments will appear again.

---

# 7. What Happens If I Train Without MLflow?

This:

```python
model.fit(X_train, y_train)
```

works normally even if MLflow is not running.

However:

```text
Model training        YES
MLflow experiment     NO
MLflow metrics        NO
MLflow parameters     NO
MLflow artifacts      NO
```

MLflow does not automatically discover models that were trained outside MLflow tracking.

---

# 8. Relationship Between Git Branches and MLflow

Git branches and MLflow are independent.

A developer can work on a feature branch:

```text
Feature branch
      |
      v
Develop model
      |
      v
Run model
      |
      v
Log MLflow experiment
      |
      v
Shared Neon database
```

The experiment becomes available to the team immediately.

The branch does **not** need to be merged into `main` first.

Likewise:

```text
Merge into main
```

does not automatically create an MLflow run.

---

# 9. Recommended Project Workflow

A recommended workflow is:

```text
Developer creates/changes model
             |
             v
Start local MLflow server
             |
             v
Run experiment
             |
             v
Log parameters + metrics
             |
             v
Compare results in MLflow
             |
             v
Commit + push code
             |
             v
Pull Request
             |
             v
Merge approved code into main
```

For important/final models:

```text
Development experiments
        |
        v
Compare models
        |
        v
Choose approach
        |
        v
Merge into main
        |
        v
Pull latest main
        |
        v
Run final model again
        |
        v
Log final MLflow run
        |
        v
Store actual trained model
```

This gives us a clean distinction:

```text
Development runs
=
experimentation and comparison

Main/final run
=
accepted project implementation
```

---

# 10. What Should Be Logged During Development?

For most experiments, log:

```text
Parameters
Metrics
Git branch
Git commit
Team member
Useful plots
Useful reports
```

Example:

```python
mlflow.log_params({
    "model": "RandomForest",
    "n_estimators": 200,
    "max_depth": 10
})

mlflow.log_metrics({
    "accuracy": 0.91,
    "f1_macro": 0.89
})
```

We do **not** need to save the actual trained model for every experiment.

---

# 11. When Should We Store the Actual Model?

Recommended policy:

```text
Development experiment
→ Parameters ✅
→ Metrics ✅
→ Tags ✅
→ Useful artifacts ✅
→ Model file usually NO


Selected/final model
→ Parameters ✅
→ Metrics ✅
→ Tags ✅
→ Artifacts ✅
→ Actual model ✅
```

This prevents unnecessary Object Storage usage.

Example:

```python
mlflow.sklearn.log_model(
    sk_model=model,
    name="model"
)
```

Only use this when the trained model itself should be preserved.

---

# 12. Recommended Git Information to Log

Important runs should include:

```text
Git branch
Git commit
Team member
```

Example:

```python
import subprocess
import mlflow


branch = subprocess.check_output(
    ["git", "branch", "--show-current"],
    text=True
).strip()


commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True
).strip()


with mlflow.start_run():

    mlflow.set_tags({
        "git_branch": branch,
        "git_commit": commit
    })
```

A run could then show:

```text
Run:
random_forest_v3

Model:
RandomForest

Accuracy:
0.91

F1:
0.89

team_member:
Alice

git_branch:
feature/model-development

git_commit:
7d248fe
```

---

# 13. Team Member Identification (optional)

It is better if each collaborator set:

```text
MLFLOW_TEAM_MEMBER=
```

inside their local `.env`.

Example:

```text
MLFLOW_TEAM_MEMBER=Alice
```

Then training code can use:

```python
import os

team_member = os.getenv(
    "MLFLOW_TEAM_MEMBER",
    "unknown"
)

mlflow.set_tag(
    "team_member",
    team_member
)
```

This allows MLflow runs to identify who performed the experiment.

Example:

```text
Run                  Team Member
---------------------------------
random_forest_v1     Alice
xgboost_v1           Bob
logistic_v2          Carol
```

---

# 14. Connecting Training Code to MLflow

Add:

```python
import mlflow

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)
```

Then select or create an experiment:

```python
mlflow.set_experiment(
    "baseline_models"
)
```

---

# 15. Example Run

```python
import mlflow

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "baseline_models"
)

with mlflow.start_run(
    run_name="random_forest_baseline"
):

    mlflow.log_param(
        "model",
        "RandomForest"
    )

    mlflow.log_param(
        "n_estimators",
        200
    )

    mlflow.log_param(
        "max_depth",
        10
    )

    mlflow.log_metric(
        "accuracy",
        0.91
    )

    mlflow.log_metric(
        "f1_macro",
        0.89
    )
```

After running, refresh:

```text
http://127.0.0.1:5000
```

The run should appear.

Because it is stored in Neon, other collaborators will also see it when they start their own MLflow servers.

---

# 16. Logging Artifacts

Artifacts are files associated with a run.

Examples:

```python
mlflow.log_artifact(
    "confusion_matrix.png"
)
```

```python
mlflow.log_artifact(
    "classification_report.csv"
)
```

The flow is:

```text
Training code
      |
      v
Local MLflow Server
      |
      v
Neon Object Storage
```

---

# 17. Suggested Experiment Organization

Use meaningful experiment names.

Avoid:

```text
test
test2
final
final2
new
final_final
```

Prefer names such as:

```text
data_preparation
baseline_models
feature_engineering
prompt_injection_detection
hyperparameter_tuning
model_comparison
final_models
```

---

# 18. Example MLflow Structure

Suppose three collaborators are testing models.

```text
Experiment: baseline_models

├── Run: logistic_regression
│   ├── team_member = Alice
│   ├── accuracy = 0.84
│   └── f1_macro = 0.82
│
├── Run: random_forest
│   ├── team_member = Bob
│   ├── accuracy = 0.90
│   └── f1_macro = 0.88
│
└── Run: xgboost
    ├── team_member = Carol
    ├── accuracy = 0.93
    └── f1_macro = 0.92
```

These development runs do not necessarily need to store trained model files.

After one approach is selected:

```text
Experiment: final_models

└── Run: final_model
    ├── git_branch = main
    ├── model_status = final
    ├── metrics
    ├── parameters
    │
    └── Stored Model
```

---

# 19. Deleting Experiments and Models

Deleting an experiment from the MLflow UI initially performs a logical/soft deletion.

Do not manually delete MLflow rows directly from Neon PostgreSQL.

Do not manually remove MLflow artifact directories from Object Storage.

Use MLflow's deletion and garbage-collection mechanisms.

For team safety:

> Do not permanently delete shared experiments or models without discussing it with the team first.

---

# 20. Important Team Rules

### Rule 1

Start MLflow before running an experiment that should be tracked.

### Rule 2

A Git push does not create an MLflow run.

### Rule 3

A merge into `main` does not automatically create or update an MLflow run.

### Rule 4

Experiments from development branches can appear in the shared MLflow UI.

### Rule 5

Tag important runs with the Git branch and Git commit.

### Rule 6

Use `MLFLOW_TEAM_MEMBER` to identify who performed a run.

### Rule 7

Do not save every development model unnecessarily.

### Rule 8

Prefer storing trained model files for selected or final models.

### Rule 9

After an important implementation is merged into `main`, preferably rerun it from `main` and log the accepted/final run.

### Rule 10

Never commit `.env` or Neon credentials.

### Rule 11

Do not delete shared experiments or models without discussing it with the team.

---
