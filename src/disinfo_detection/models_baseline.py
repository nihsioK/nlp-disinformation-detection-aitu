"""Baseline TF-IDF models for LIAR truthfulness classification."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import yaml
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


VALID_CLASSIFIERS = {"svm", "naive_bayes", "random_forest"}


def load_baseline_config(config_path: str = "config/baseline.yml") -> dict:
    """Load the baseline YAML configuration.

    Args:
        config_path: Path to the baseline configuration file.

    Returns:
        Parsed baseline configuration.
    """

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset_config(config_path: str = "config/dataset.yaml") -> dict:
    """Load the dataset YAML configuration.

    Args:
        config_path: Path to the dataset configuration file.

    Returns:
        Parsed dataset configuration.
    """

    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class TFIDFBaseline:
    """Train and serve a TF-IDF plus classical-ML baseline.

    Args:
        classifier_type: One of `svm`, `naive_bayes`, or `random_forest`.
        config_path: Path to the baseline configuration.
        dataset_config_path: Path to the dataset configuration.
    """

    def __init__(
        self,
        classifier_type: str,
        config_path: str = "config/baseline.yml",
        dataset_config_path: str = "config/dataset.yaml",
    ) -> None:
        if classifier_type not in VALID_CLASSIFIERS:
            raise ValueError(f"classifier_type must be one of {sorted(VALID_CLASSIFIERS)}")

        self.classifier_type = classifier_type
        self.config_path = config_path
        self.dataset_config_path = dataset_config_path
        self.config = load_baseline_config(config_path)
        self.dataset_config = load_dataset_config(dataset_config_path)
        self.label_names = self.dataset_config["liar"]["label_names"]
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Build the sklearn pipeline from the current config.

        Returns:
            Configured TF-IDF pipeline.
        """

        tfidf_cfg = self.config["tfidf"]
        vectorizer = TfidfVectorizer(
            max_features=tfidf_cfg["max_features"],
            ngram_range=tuple(tfidf_cfg["ngram_range"]),
            min_df=tfidf_cfg["min_df"],
            sublinear_tf=tfidf_cfg["sublinear_tf"],
        )
        classifier = self._build_classifier()
        return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])

    def _build_classifier(self):
        """Build the configured sklearn classifier.

        Returns:
            Instantiated sklearn classifier.
        """

        if self.classifier_type == "svm":
            svm_cfg = self.config["svm"]
            return LinearSVC(
                C=svm_cfg["C"],
                class_weight=svm_cfg["class_weight"],
                max_iter=svm_cfg["max_iter"],
            )

        if self.classifier_type == "naive_bayes":
            nb_cfg = self.config["naive_bayes"]
            return MultinomialNB(alpha=nb_cfg["alpha"])

        rf_cfg = self.config["random_forest"]
        return RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_leaf=rf_cfg["min_samples_leaf"],
            class_weight=rf_cfg["class_weight"],
            n_jobs=rf_cfg["n_jobs"],
            random_state=rf_cfg["random_state"],
        )

    def fit(self, X_train: list[str], y_train: list[int]) -> "TFIDFBaseline":
        """Fit the baseline pipeline.

        Args:
            X_train: Training texts.
            y_train: Integer label ids.

        Returns:
            The fitted model instance.
        """

        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: list[str]) -> list[int]:
        """Predict label ids for input texts.

        Args:
            X: Input texts.

        Returns:
            Predicted label ids.
        """

        predictions = self.pipeline.predict(X)
        return [int(prediction) for prediction in predictions.tolist()]

    def predict_proba(self, X: list[str]) -> np.ndarray:
        """Estimate class probabilities or probability-like scores.

        Args:
            X: Input texts.

        Returns:
            Dense array of class probabilities.
        """

        classifier = self.pipeline.named_steps["classifier"]
        if hasattr(classifier, "predict_proba"):
            return self.pipeline.predict_proba(X)

        decision_scores = self.pipeline.decision_function(X)
        if decision_scores.ndim == 1:
            decision_scores = np.vstack([-decision_scores, decision_scores]).T
        return softmax(decision_scores, axis=1)

    def get_top_features(self, n: int = 20) -> dict[str, list[tuple[str, float]]]:
        """Return top TF-IDF features for random forest models.

        Args:
            n: Number of top features to return.

        Returns:
            Dictionary containing global top features under the `global` key.

        Raises:
            ValueError: If the classifier is not random forest or is unfitted.
        """

        if self.classifier_type != "random_forest":
            raise ValueError("Top feature inspection is only available for random_forest.")

        vectorizer = self.pipeline.named_steps["tfidf"]
        classifier = self.pipeline.named_steps["classifier"]
        if not hasattr(classifier, "feature_importances_"):
            raise ValueError("The random forest model must be fitted before reading feature importances.")

        feature_names = vectorizer.get_feature_names_out()
        importances = classifier.feature_importances_
        top_indices = np.argsort(importances)[::-1][:n]
        features = [(feature_names[index], float(importances[index])) for index in top_indices]
        return {"global": features}

    def save(self, path: str) -> None:
        """Persist the full baseline object with joblib.

        Args:
            path: Destination file path.
        """

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, path: str) -> "TFIDFBaseline":
        """Load a persisted baseline object.

        Args:
            path: Joblib path produced by `save()`.

        Returns:
            Deserialized baseline model.
        """

        return joblib.load(path)
