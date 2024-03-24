import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        try:
            cols_to_drop = ["customer_zip_code_prefix"]
            X = data.drop(columns=[cols_to_drop])
            print(X.columns)
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e


class DataSplitStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    