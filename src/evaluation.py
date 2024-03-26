import logging
from abc import ABC, abstractmethod

from sklearn.metrics import r2_score
import numpy as np


class EvaluationStrategy(ABC):
    """Abstract class to evaluate our models."""
    @abstractmethod
    def calculate_score(self, y_true, y_pred):
        """
        Calculate scores for the model.
        """
        pass


class MSE(EvaluationStrategy):
    """Mean Squared Error."""

    def calculate_score(self, y_true, y_pred):
        """
        Calculate MSE.
        """
        try:
            logging.info(f"Calculating MSE for model {model}")
            mse = ((y_true - y_pred) ** 2).mean()
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE: {e}")
            raise e


class R2(EvaluationStrategy):
    """R2 Score."""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate R2.
        """
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2: {e}")
            raise e


class RMSE(EvaluationStrategy):
    """Root Mean Squared Error."""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate RMSE.
        """
        try:
            rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise e
