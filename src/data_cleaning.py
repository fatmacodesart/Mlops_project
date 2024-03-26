import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union, Tuple


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:

        try:
            data = data.select_dtypes(include=[np.number])
            data['product_weight_g'] = data['product_weight_g'].ffill()
            data['product_length_cm'] = data['product_length_cm'].ffill()
            data['product_height_cm'] = data['product_height_cm'].ffill()
            data['product_width_cm'] = data['product_width_cm'].ffill()

            cols_to_drop = ["customer_zip_code_prefix",
                            "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e


class DataSplitStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in data splitting: {e}")
            raise e


class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in data cleaning: {e}")
            raise e
