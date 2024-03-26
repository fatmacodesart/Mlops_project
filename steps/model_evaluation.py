import logging
from typing import Tuple

from zenml import step
import pandas as pd
from typing_extensions import Annotated

from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin


@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   ) -> Tuple[Annotated[float, "R2 Score"],
                              Annotated[float, "RMSE"],
                              ]:
    """Evaluate the model."""
    try:
        predictions = model.predict(X_test)
        mse = MSE()
        mse_score = mse.calculate_score(y_test, predictions)

        r2 = R2()
        r2_score = r2.calculate_score(y_test, predictions)

        rmse = RMSE()
        rmse_score = rmse.calculate_score(y_test, predictions)

        return r2_score, rmse

    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        return
