from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.data_cleaning import clean_data
from steps.model_training import train_model
from steps.model_evaluation import evaluate_model


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    """Simple training pipeline"""
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    trained_model = train_model(X_train, y_train)
    r2_score , rmse_score = evaluate_model(trained_model, X_test, y_test)
