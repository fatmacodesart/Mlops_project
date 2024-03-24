from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.data_cleaning import clean_data
from steps.model_training import train_model
from steps.model_evaluation import evaluate_model

@pipeline(enable_cache=True)

def train_pipeline(data_path: str):
    """Simple training pipeline"""
    df = ingest_data(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)