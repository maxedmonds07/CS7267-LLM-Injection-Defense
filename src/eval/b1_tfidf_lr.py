import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats

# Prepare data using the existing df in kernel
df_eval = df.dropna(subset=['label', 'text']).copy()
df_eval['label'] = df_eval['label'].astype(int)

# Keep attack families grouped, but assign each benign (NaN) row a unique group ID to distribute safe texts
df_eval['cv_group'] = df_eval['attack_family'].fillna(pd.Series([f'benign_group_{i}' for i in range(len(df_eval))], index=df_eval.index))

# Setup Family-Grouped Cross-Validation
gkf = GroupKFold(n_splits=5)
X = df_eval['text'].values
y = df_eval['label'].values
groups = df_eval['cv_group'].values

auc_scores = []
tpr_scores = []

print("\n--- Running 5-Fold Family-Grouped Cross-Validation ---")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Pipeline: TF-IDF + Logistic Regression
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)
    
    # Predict probabilities for validation fold
    y_pred_proba = clf.predict_proba(X_test_vec)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc)
    
    # Calculate TPR at 5% FPR
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    idx_5fpr = np.argmin(np.abs(fpr - 0.05))
    tpr_at_5fpr = tpr[idx_5fpr]
    tpr_scores.append(tpr_at_5fpr)
    
    print(f"Fold {fold+1}: AUROC = {auc:.4f}, TPR @ 5% FPR = {tpr_at_5fpr:.4f}")

# Calculate mean and 95% Confidence Intervals
def print_metric_summary(name, scores):
    mean = np.mean(scores)
    sem = stats.sem(scores)
    low, high = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=sem)
    print(f"{name}: {mean:.4f} (95% CI: {low:.4f} - {high:.4f})")

print("\n--- Overall Results (Mean \u00b1 95% CI) ---")
print_metric_summary("AUROC", auc_scores)
print_metric_summary("TPR @ 5% FPR", tpr_scores)

# Length-Binned AUROC
print("\n--- Calculating Length-Binned AUROC ---")
df_eval['text_length'] = df_eval['text'].apply(len)
df_eval['length_bin'] = pd.qcut(df_eval['text_length'], q=3, labels=['Short', 'Medium', 'Long'])

# Train full model to analyze shortcut learning
vectorizer_full = TfidfVectorizer(max_features=5000, stop_words='english')
X_full_vec = vectorizer_full.fit_transform(df_eval['text'])
clf_full = LogisticRegression(max_iter=1000, class_weight='balanced')
clf_full.fit(X_full_vec, df_eval['label'])
df_eval['pred_proba'] = clf_full.predict_proba(X_full_vec)[:, 1]

for bin_name in ['Short', 'Medium', 'Long']:
    bin_df = df_eval[df_eval['length_bin'] == bin_name]
    if len(bin_df['label'].unique()) > 1:
        bin_auc = roc_auc_score(bin_df['label'], bin_df['pred_proba'])
        print(f"{bin_name} texts (n={len(bin_df)}): AUROC = {bin_auc:.4f}")
    else:
        print(f"{bin_name} texts (n={len(bin_df)}): Single-class only, cannot compute AUROC.")
