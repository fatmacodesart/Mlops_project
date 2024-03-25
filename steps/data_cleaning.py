import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated

from src.data_cleaning import DataPreprocessStrategy, DataSplitStrategy, DataCleaning 

@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """Cleans data"""
    
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(preprocessed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(f"Data cleaning completed. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        raise e
