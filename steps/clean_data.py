import logging
import pandas as pd
from zenml import step

@step
def clean_df(data: pd.DataFrame) -> None:
    """Cleans data"""
    pass