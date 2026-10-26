"""TF-IDF detector: logistic regression on word 1-2 grams and character 3-5 grams.

Fast and inspectable, top_features shows whether it learned injection wording or source artifacts.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline


class TfidfDetector:
    """Logistic regression on TF-IDF word and character n-grams."""

    name = "tfidf"

    def __init__(self, C=1.0, min_df=2):
        self.C = C
        self.pipeline = make_pipeline(
            FeatureUnion(
                [
                    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
                    # char_wb keeps n-grams inside word boundaries, so "ign" and "gnore" survive
                    # obfuscations such as extra spaces between words.
                    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min_df,
                                             sublinear_tf=True)),
                ]
            ),
            LogisticRegression(C=C, max_iter=2000),
        )

    def fit(self, texts, labels):
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts):
        """Probability that each text is adversarial, one value per text."""
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def top_features(self, n=20):
        """The n features pushing hardest toward each label, as "word_..." or "char_..."."""
        union, model = self.pipeline.steps[0][1], self.pipeline.steps[1][1]
        names = union.get_feature_names_out()
        order = np.argsort(model.coef_[0])
        return {"adversarial": list(names[order[::-1][:n]]), "benign": list(names[order[:n]])}
    