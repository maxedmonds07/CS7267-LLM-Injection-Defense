import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Ensure repository root is in sys.path to load src successfully
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.eval.family_cv import family_folds
from src.eval.metrics import tpr_at_fpr, bootstrap_ci, length_binned_auroc

def run_evaluation(data_path='data/processed/dataset.parquet'):
    # Load data
    df = pd.read_parquet(data_path)

    # Filter to train_pool
    df_eval = df[df['role'] == 'train_pool'].dropna(subset=['label', 'text']).copy()
    df_eval['label'] = df_eval['label'].astype(int)

    # Cross-validation using family_folds
    folds = family_folds(df_eval, k=5, seed=7267)

    X = df_eval['text'].values
    y = df_eval['label'].values
    oof_preds = np.zeros(len(df_eval))

    print('\n--- Running 5-Fold Family-Grouped Cross-Validation ---')
    for fold in range(5):
        train_idx = np.where(folds != fold)[0]
        test_idx = np.where(folds == fold)[0]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Pipeline
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train_vec, y_train)

        y_pred_proba = clf.predict_proba(X_test_vec)[:, 1]
        oof_preds[test_idx] = y_pred_proba

        # Calculate fold metrics
        fold_auc = roc_auc_score(y_test, y_pred_proba)
        fold_tpr = tpr_at_fpr(y_pred_proba, y_test, fpr=0.05)
        print(f'Fold {fold+1}: AUROC = {fold_auc:.4f}, TPR @ 5% FPR = {fold_tpr:.4f}')

    df_eval['oof_pred_proba'] = oof_preds

    # Bootstrap CI
    print('\n--- Overall Validation Results (95% Bootstrap CI) ---')

    res_auroc = bootstrap_ci(
        df_eval,
        lambda d: roc_auc_score(d['label'].values, d['oof_pred_proba'].values)
    )
    res_tpr = bootstrap_ci(
        df_eval,
        lambda d: tpr_at_fpr(d['oof_pred_proba'].values, d['label'].values, 0.05)
    )

    # Unpack CI results safely
    def unpack(res):
        if isinstance(res, tuple):
            if len(res) == 2:
                mean, ci = res
                if isinstance(ci, (list, tuple, np.ndarray)):
                    return mean, ci[0], ci[1]
                else:
                    return mean, mean - ci, mean + ci
            elif len(res) == 3:
                return res
        return res, 0.0, 0.0

    auroc_mean, auroc_low, auroc_high = unpack(res_auroc)
    tpr_mean, tpr_low, tpr_high = unpack(res_tpr)

    print(f'AUROC: {auroc_mean:.4f} (95% CI: {auroc_low:.4f} - {auroc_high:.4f})')
    print(f'TPR @ 5% FPR: {tpr_mean:.4f} (95% CI: {tpr_low:.4f} - {tpr_high:.4f})')

    # Length-binned AUROC on OOF Predictions
    print('\n--- Out-Of-Fold Length-Binned AUROC ---')
    binned_score = length_binned_auroc(
        df_eval['oof_pred_proba'].values,
        df_eval['label'].values,
        df_eval['text'].values
    )
    print(f'Averaged Length-Binned AUROC: {binned_score:.4f}')

if __name__ == "__main__":
    run_evaluation()
