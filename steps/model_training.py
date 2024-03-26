import logging

import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin #for reg output

from .config import ModelNameConfig

@step
def train_model(X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                config: ModelNameConfig ) -> RegressorMixin:
    """Trains a model and returns regressor output for evaluation."""
    model = None
    if config.model_name == 'LinearRegression':
        model = LinearRegressionModel()
        trained_model = model.train(X_train, y_train)
        return trained_model
    else:
        raise ValueError(f"Model {config.model_name} not supported.") 