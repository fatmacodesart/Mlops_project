import logging

import pandas as pd
from zenml import step

class IngestData:
    """Reads data from a given path"""
    def __init__(self,data_path: str):
        self.data_path = data_path
    def get_data(self):
        logging.info(f"Reading data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """Ingests data from a given path"""
    try:
        reader = IngestData(data_path)
        df = reader.get_data()
        return df
    except Exception as e:
        logging.error(f"Error reading data from {data_path}")
        logging.error(e)
        raise e