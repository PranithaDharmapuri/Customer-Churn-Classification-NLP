from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class RandomObliqueForestTrees(BaseEstimator, ClassifierMixin):
    """
   RandomObliqueForestTrees (ROFT)
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        # Build internal EOT model
        self.eot_model = self.build_eot()

        self.rule_list_ = None

    def build_eot(self):

        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.eot_model.fit(X, y)
        return self

    def predict(self, X):
        return self.eot_model.predict(X)

    def predict_proba(self, X):
        return self.eot_model.predict_proba(X)

    def _generate_rule_list(self):
        """
        Extract light interpretable EOT rules from tree estimators.
        Uses first 5 trees for readability.
        """
        rules = []
        for estimator in self.eot_model.estimators_[:5]:
            tree = estimator.tree_
            feature = tree.feature
            rules.append(
                f"EOT Rule → Tree depth={tree.max_depth}, "
                f"Features used={np.unique(feature[feature >= 0]).size}"
            )
        self.rule_list_ = rules

    def get_rule_list(self):
        if self.rule_list_ is None:
            raise ValueError("Model must be fitted before extracting EOT rules.")
        return self.rule_list_

    def score(self, X, y):
        return self.eot_model.score(X, y)
