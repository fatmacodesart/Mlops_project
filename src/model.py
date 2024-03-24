import logging
from abc import ABC, abstractmethod

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

    @abstractmethod
    def predict(self, X_test):
        pass