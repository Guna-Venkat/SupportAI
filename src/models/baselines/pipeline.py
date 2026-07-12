"""
pipeline.py
===========
Classical machine learning pipelines for SupportAI.

Provides utility functions to instantiate, train, and test classical baselines
(Logistic Regression, Linear SVM, and Multinomial Naive Bayes) using TF-IDF text features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def create_baseline_pipeline(
    model_type: str,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    seed: int = 42,
    c_val: float = 1.0,
    alpha: float = 1.0,
) -> Pipeline:
    """Instantiates a scikit-learn Pipeline with a TF-IDF vectorizer and classifier.

    Args:
        model_type: One of 'logistic_regression', 'linear_svm', or 'naive_bayes'.
        max_features: Maximum vocabulary size for TF-IDF.
        ngram_range: Token n-gram size ranges.
        seed: Random state seed.
        c_val: Regularization parameter for LogisticRegression and LinearSVC.
        alpha: Smoothing parameter for MultinomialNB.

    Returns:
        Configured scikit-learn Pipeline instance.

    Raises:
        ValueError: If an unsupported model_type is provided.
    """
    logger.info(
        "Creating pipeline for '%s' | max_features=%d | ngram_range=%s",
        model_type,
        max_features,
        ngram_range,
    )

    # 1. Instantiate TF-IDF Feature Extractor
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    # 2. Select Classifier
    if model_type == "logistic_regression":
        classifier = LogisticRegression(C=c_val, max_iter=1000, random_state=seed, solver="lbfgs")
    elif model_type == "linear_svm":
        classifier = LinearSVC(C=c_val, random_state=seed, dual="auto")
    elif model_type == "naive_bayes":
        classifier = MultinomialNB(alpha=alpha)
    else:
        msg = f"Unsupported baseline model type: {model_type}"
        logger.error(msg)
        raise ValueError(msg)

    # 3. Build scikit-learn Pipeline
    pipeline = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    return pipeline
