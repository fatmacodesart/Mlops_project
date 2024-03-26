import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """Abstract class for all models."""
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model.
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        """
        pass

class LinearRegressionModel(Model):
    """Regression model class."""
    def train(self, X_train, y_train, **kwargs):
        """Train the regression model.
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
            **kwargs: Additional keyword arguments
        """
        try:
            lin_reg = LinearRegression(**kwargs)
            lin_reg.fit(X_train, y_train)
            logging.info("Model training completed.")
            return lin_reg
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise e