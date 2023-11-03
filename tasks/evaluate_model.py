from src.model_evaluation import ModelEvaluation, F1ScoreEvaluationStrategy, AccuracyEvaluationStrategy, PrecisionEvaluationStrategy, RecallEvaluationStrategy, ConfusionMatrixEvaluationStrategy
from prefect import task
import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

@task
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[
    Annotated[float, "accuracy"],
    Annotated[float, "precision"],
    Annotated[float, "recall"],
]:
   """
    This function evaluates a machine learning model using multiple evaluation strategies and logs the results using MLflow.

    Args:
        model: The trained machine learning model to be evaluated.
        X_test (pd.DataFrame): The feature DataFrame for testing.
        y_test (pd.Series): The true label Series for testing.

    Returns:
        evaluation metrics (accuracy, precision, recall).

    Raises:
        None.
    """
   y_pred = model.predict(X_test)
   accuracy_strategy = AccuracyEvaluationStrategy()
   precision_strategy = PrecisionEvaluationStrategy()
   recall_strategy = RecallEvaluationStrategy()
   f1_score_strategy = F1ScoreEvaluationStrategy()
   confusion_matrix_strategy = ConfusionMatrixEvaluationStrategy()
   evaluation = ModelEvaluation(accuracy_strategy)
   accuracy = evaluation.evaluate_model(y_test, y_pred)
   evaluation.strategy = precision_strategy
   precision = evaluation.evaluate_model(y_test, y_pred)
   evaluation.strategy = recall_strategy
   recall = evaluation.evaluate_model(y_test, y_pred)
   evaluation.strategy = f1_score_strategy
   f1_score= evaluation.evaluate_model(y_test, y_pred)
   evaluation.strategy = confusion_matrix_strategy
   confusion_matrix = evaluation.evaluate_model(y_test, y_pred)
   return accuracy, precision, recall