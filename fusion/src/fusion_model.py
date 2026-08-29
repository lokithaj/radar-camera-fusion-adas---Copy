from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class FusionMLModel:
    """
    Load and run the trained radar-camera association model.
    """

    def __init__(self, model_path):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Fusion model not found: {model_path}"
            )

        bundle = joblib.load(
            model_path
        )

        if not isinstance(bundle, dict):
            raise ValueError(
                "Invalid fusion model file."
            )

        if "model" not in bundle:
            raise ValueError(
                "Model file does not contain 'model'."
            )

        if "features" not in bundle:
            raise ValueError(
                "Model file does not contain 'features'."
            )

        self.model = bundle["model"]
        self.features = bundle["features"]

    def predict(self, features):
        """
        Predict whether a radar-camera pair matches.

        Parameters
        ----------
        features : dict
            Feature dictionary using the same feature names
            used during training.

        Returns
        -------
        int
            1 = MATCH
            0 = NO MATCH
        """

        row = {
            feature: features.get(
                feature,
                0.0,
            )
            for feature in self.features
        }

        df = pd.DataFrame(
            [row],
            columns=self.features,
        )

        prediction = self.model.predict(
            df
        )[0]

        return int(prediction)

    def predict_probability(self, features):
        """
        Return the model's probability of MATCH.
        """

        row = {
            feature: features.get(
                feature,
                0.0,
            )
            for feature in self.features
        }

        df = pd.DataFrame(
            [row],
            columns=self.features,
        )

        probabilities = (
            self.model.predict_proba(df)[0]
        )

        classes = list(
            self.model.classes_
        )

        if 1 in classes:
            match_index = classes.index(1)
            return float(
                probabilities[match_index]
            )

        return 0.0