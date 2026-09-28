# Shared MLflow Setup

## CS7267 – LLM Injection Defense Project

This project uses **MLflow** to centrally track machine-learning experiments performed by different team members.

We do **not** maintain one continuously running cloud MLflow server.

Instead:

- Each team member runs an MLflow server locally on their own computer.
- All local MLflow servers connect to the **same Neon PostgreSQL database**.
- All local MLflow servers connect to the **same Neon Object Storage bucket**.
- Therefore, everyone sees the same experiments, runs, parameters, metrics, and artifacts.
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
                MLflow experiments
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
                              |
                              v
                  Neon Object Storage
                       FREE TIER
                ------------------------
                Models
                Plots
                CSV files
                Reports
                Other MLflow artifacts

                Free-plan allowance:
                • 5 GB object storage/project
```

As of September 2026, Neon lists **0.5 GB of Postgres storage and 100 CU-hours per project**, as well as 10 branches per project, on its Free plan. Neon also includes **5 GB of Object Storage per project** on the Free plan.

For a university term project, these limits should generally be sufficient as long as we do not unnecessarily log very large model files or datasets.

---

# 2. What Each Component Does

## GitHub

GitHub tracks our **code**.

Examples:

```text
Python scripts
Jupyter notebooks
Model implementations
Preprocessing code
Configuration
Documentation
Requirements
```

GitHub provides:

```text
Branches
Commits
Pull Requests
Merge history
Version control
```

---

## MLflow

MLflow tracks our **machine-learning experiments**.

Examples:

```text
Model name
Hyperparameters
Accuracy
Precision
Recall
F1-score
AUROC
Training configuration
Git commit
Git branch
Plots
Model files
Other artifacts
```

MLflow allows us to answer questions such as:

```text
Which model performed best?

What hyperparameters were used?

Who ran this experiment?

Which version of the code produced this result?

What metrics did a previous experiment achieve?
```

---

## Neon PostgreSQL

Neon PostgreSQL is the shared database behind MLflow.

It contains MLflow metadata such as:

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

That is why everyone can see the same experiments.

---

## Neon Object Storage

Neon Object Storage stores MLflow files and artifacts.

Examples:

```text
Trained models
Confusion matrices
ROC curves
PR curves
SHAP plots
Feature-importance plots
Classification reports
CSV files
JSON files
Other experiment artifacts
```

All team members use the same object-storage bucket.

---

# 3. How MLflow Actually Works

An important point:

## MLflow does NOT automatically track GitHub activity.

The following:

```text
git add
git commit
git push
Pull Request
Merge to main
```

does **not** automatically create an MLflow experiment or run.

MLflow records information only when your Python code actually communicates with MLflow.

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

When this code executes:

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
       +---- parameters
       +---- metrics
       +---- experiment
       +---- run information
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

# 4. Do I Need to Run the MLflow Server Every Time?

## Yes, when you are actively logging or viewing MLflow data.

Before running code that contains:

```python
mlflow.start_run()
mlflow.log_param()
mlflow.log_metric()
mlflow.log_artifact()
mlflow.sklearn.log_model()
```

start your local MLflow server.

From:

```text
mlflow-server/
```

run:

```bash
python start_mlflow.py
```

Keep that terminal open while you are running experiments.

Your training code communicates with:

```text
http://127.0.0.1:5000
```

---

## You do NOT need to leave the server running continuously.

Once an experiment has been logged:

```text
Experiment
    |
    v
Neon database / storage
```

the information is persistent.

You can stop MLflow using:

```text
Ctrl + C
```

Nothing is deleted.

Later, run:

```bash
python start_mlflow.py
```

again.

Open:

```text
http://127.0.0.1:5000
```

and all previously logged experiments will appear again.

---

# 5. What Happens If I Train a Model Without Starting MLflow?

Suppose you run:

```python
model.fit(X_train, y_train)
```

but you never start MLflow or never execute MLflow logging commands.

The model can still train normally.

However:

```text
Model training        YES

Git code              unchanged

MLflow experiment     NO

MLflow metrics        NO

MLflow parameters     NO
```

MLflow does not automatically discover models that were trained elsewhere.

Therefore, if an experiment is important and should be available to the team, run it while MLflow is configured and log the relevant information.

---

# 6. Relationship Between Git Branches and MLflow

Git branches and MLflow experiments are independent.

For example:

```text
Developer working on a branch
            |
            v
Runs model
            |
            v
Logs experiment to MLflow
            |
            v
Shared Neon database
```

That experiment immediately becomes available to the rest of the team.

The code does **not** need to be merged into `main` before the experiment can be logged.

Likewise, merging code into `main` does not automatically create an MLflow run.

---

# 7. What Happens When Code Is Merged Into `main`?

Suppose a developer:

```text
1. Develops Model A

2. Runs Model A

3. Logs Model A to MLflow

4. Commits the code

5. Opens a Pull Request

6. Merges the code into main
```

The MLflow experiment created in Step 3 remains exactly where it is.

The Git merge does not move or duplicate the experiment.

Conceptually:

```text
                        GitHub
                          |
Feature code ------------+
                          |
                          v
                         main


                       MLflow
                          |
                          v
                    Existing Run A
                    Existing Run B
                    Existing Run C
```

These systems are separate.

---

# 8. Does MLflow Show Only Models From `main`?

## No.

By default, MLflow shows **all runs stored in the shared Neon database**.

That can include experiments performed from:

```text
Development branches
Feature branches
main
Old experiments
Failed experiments
Hyperparameter trials
Final experiments
```

MLflow does not automatically decide:

> This code is now merged into main, therefore only show this model.

Instead, we should identify where important runs came from using tags.

---

# 9. Recommended Workflow

A good workflow for this project is:

```text
Developer writes/changes model code
             |
             v
Start local MLflow server
             |
             v
Run experiment
             |
             v
Log metrics/parameters/artifacts
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

After a model has been accepted into `main`, we can optionally run the accepted model **once again from `main`**.

That gives us a clean final MLflow run representing the actual code currently in `main`.

Recommended final workflow:

```text
Development run
      |
      v
MLflow comparison
      |
      v
Choose model
      |
      v
Merge code into main
      |
      v
Run model from main
      |
      v
Create final MLflow run
```

This final run can be tagged:

```text
source_branch = main
model_status = final
```

That makes it easy to distinguish final/accepted models from experimental models.

---

# 10. Recommended Git Information to Log

Every important MLflow run should include:

```text
Git branch
Git commit
Team member
```

This makes experiments reproducible.

For example:

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

    mlflow.set_tag(
        "git_branch",
        branch
    )

    mlflow.set_tag(
        "git_commit",
        commit
    )
```

The resulting MLflow run could contain:

```text
Run:
random_forest_v3

Model:
RandomForest

Accuracy:
0.91

F1:
0.89

git_branch:
feature/model-development

git_commit:
7d248fe...

team_member:
Member 2
```

After code is merged and rerun from `main`:

```text
git_branch:
main

git_commit:
85a2fb4...

model_status:
final
```

---

# 11. First-Time Setup

Each collaborator performs these steps once.

## Clone the Repository

```bash
git clone https://github.com/maxedmonds07/CS7267-LLM-Injection-Defense.git
```

Enter the repository:

```bash
cd CS7267-LLM-Injection-Defense
```

Enter:

```bash
cd mlflow-server
```

---

# 12. Create a Virtual Environment

## Windows

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## macOS / Linux

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

---

# 13. Install MLflow Requirements

```bash
pip install -r requirements.txt
```

The project uses the same requirements file so team members use compatible MLflow versions.

Do not independently upgrade MLflow unless the change is made through the repository's `requirements.txt`.

---

# 14. Configure the `.env` File

Each team member needs:

```text
mlflow-server/.env
```

Copy:

```text
.env.example
```

and populate it with the shared credentials supplied by the project owner.

Required settings include:

```text
DATABASE_URL

MLFLOW_ARTIFACT_BUCKET

AWS_ACCESS_KEY_ID

AWS_SECRET_ACCESS_KEY

MLFLOW_S3_ENDPOINT_URL
```

All team members use the same:

```text
Neon PostgreSQL database

Neon Object Storage bucket
```

This is what makes MLflow collaborative.

---

# 15. Never Commit `.env`

The `.env` file contains credentials and must never be pushed to GitHub.

Make sure:

```text
.env
```

is listed in `.gitignore`.

Never place database passwords or storage credentials inside:

```text
Python scripts
Jupyter notebooks
README files
Git commits
Pull Requests
GitHub Issues
```

---

# 16. Starting MLflow

Every time you want to work with MLflow:

### Step 1

Open a terminal.

### Step 2

Enter:

```bash
cd mlflow-server
```

### Step 3

Activate the virtual environment.

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 4

Start MLflow:

```bash
python start_mlflow.py
```

### Step 5

Open:

```text
http://127.0.0.1:5000
```

Keep the MLflow terminal open while running experiments.

---

# 17. Connecting Training Code to MLflow

Your training code should contain:

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

# 18. Example Run

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

Refresh:

```text
http://127.0.0.1:5000
```

and the run should appear.

Because it is stored in Neon, other collaborators will also see it when they start their own local MLflow servers.

---

# 19. Logging Artifacts

For example:

```python
mlflow.log_artifact(
    "confusion_matrix.png"
)
```

or:

```python
mlflow.log_artifact(
    "classification_report.csv"
)
```

The file is sent:

```text
Training code
      |
      v
Local MLflow Server
      |
      v
Neon Object Storage
```

Other team members can then access it from their MLflow UI.

---

# 20. Logging Models

Example for Scikit-Learn:

```python
import mlflow.sklearn

mlflow.sklearn.log_model(
    model,
    name="model"
)
```

The run metadata is stored in PostgreSQL while the model files are stored in Object Storage.

---

# 21. Suggested Experiment Organization

Use clear experiment names rather than names such as:

```text
test
test2
final
final2
```

For example:

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

# 22. Important Team Rules

### Rule 1

Start MLflow before running an experiment that should be tracked.

### Rule 2

A Git push does not create an MLflow run.

### Rule 3

A merge into `main` does not create or update an MLflow run.

### Rule 4

Experiments performed on development branches can still appear in the shared MLflow UI.

### Rule 5

Tag important runs with their Git branch and Git commit.

### Rule 6

After an important model is merged into `main`, preferably run it again from `main` and record that as the final/accepted run.

### Rule 7

Do not delete shared MLflow experiments or models without discussing it with the team.

### Rule 8

Never commit Neon credentials or the `.env` file.

---

# 23. Typical Development Example

Suppose a team member develops a new model.

```text
1. Start local MLflow

2. Develop/train model

3. Log experiment to MLflow

4. Compare metrics with existing experiments

5. Commit model code

6. Push code to GitHub

7. Create Pull Request

8. Merge approved code into main

9. Pull latest main

10. Optionally rerun the accepted model from main

11. Log the main run as the final model
```

The final experiment could contain:

```text
model = RandomForest

accuracy = 0.91

f1_macro = 0.89

git_branch = main

git_commit = 93ad281...

model_status = final
```

This gives us both:

```text
GitHub
→ authoritative final code

MLflow
→ authoritative experiment/model results
```

---

# 24. Daily Quick Start

For normal work:

```bash
git pull
```

Then:

```bash
cd mlflow-server
```

Activate the environment:

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Start:

```bash
python start_mlflow.py
```

Open:

```text
http://127.0.0.1:5000
```

Then run training code that points to:

```python
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)
```

When finished:

```text
Ctrl + C
```

The server stops, but all shared MLflow data remains safely stored in Neon.

---

# 25. Final Mental Model

The easiest way to understand our setup is:

```text
GitHub
=
Where is the code?


MLflow
=
What happened when we ran the code?


Neon PostgreSQL
=
Shared experiment records


Neon Object Storage
=
Shared model/artifact files
```

And most importantly:

```text
Git push
        X
        |
        v
Does NOT automatically update MLflow


Running code with MLflow logging
        |
        v
DOES update MLflow
```

The local MLflow server only needs to be running while you are actively using MLflow.

Because all persistent data lives in Neon, each collaborator can stop and restart their local MLflow server whenever needed without losing shared experiment history.
