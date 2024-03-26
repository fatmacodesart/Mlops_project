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
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans data and returns training and testing data.
    Args:
        data (pd.DataFrame): Input data
    Returns:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Testing data
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
    """

    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(preprocessed_data, split_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(
            f"Data cleaning completed.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data cleaning: {e}")
        raise e
