import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        try:
            data = data.select_dtypes(include=[np.number])
            data['product_weight_g']= data['product_weight_g'].ffill()
            data['product_length_cm'] = data['product_length_cm'].ffill()
            data['product_height_cm'] = data['product_height_cm'].ffill()
            data['product_width_cm'] = data['product_width_cm'].ffill()
            
            cols_to_drop = ["customer_zip_code_prefix",
                            "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            
        except Exception as e:
            logging.error(f"Error in data preprocessing: {e}")
            raise e


class DataSplitStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
