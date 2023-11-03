from abc import ABC, abstractmethod
import logging
import numpy as np
from typing import Union
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define an abstract base class for model evaluation strategies
class ModelEvaluationStrategy(ABC):
    """
    This abstract base class defines a common interface for model evaluation strategies.

    Attributes:
        None.

    Methods:
        - evaluate: It evaluates a model's performance based on the true labels and predicted labels and returns a floating-point score.
    """
    @abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        This abstract method must be implemented by concrete strategy classes for model evaluation.

        Args:
            y_true (pd.Series): true labels.
            y_pred (np.ndarray): predicted labels.

       Returns:
            float: The evaluation score based on the model's performance.
        """
        pass

# Concrete strategy for accuracy
class AccuracyEvaluationStrategy(ModelEvaluationStrategy):
    """
    This concrete strategy evaluates a model's performance using accuracy.

    Attributes:
        None.

    Methods:
        - evaluate: This method calculates and returns the accuracy score based on true and predicted labels.
    """
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Evaluates a model's performance using accuracy.

        Args:
            y_true (pd.Series): true labels.
            y_pred (np.ndarray): predicted labels.

        Returns:
            accuracy score (float): the proportion of correctly predicted labels.
        """
        try:
            return accuracy_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error in accuracy evaluation: {e}")
            raise e

# Concrete strategy for precision
class PrecisionEvaluationStrategy(ModelEvaluationStrategy):
    """
    This concrete strategy evaluates a model's performance using precision.

    Attributes:
        None.

    Methods:
        - evaluate: This method calculates and returns the precision score based on true and predicted labels.
    """
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Evaluates a model's performance using precision.

        Args:
            y_true (pd.Series): true labels.
            y_pred (np.ndarray): predicted labels.

        Returns:
            precision score (float): indicating the proportion of true positive predictions.
        """
        try:
            return precision_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error in precision evaluation: {e}")
            raise e

# Concrete strategy for recall
class RecallEvaluationStrategy(ModelEvaluationStrategy):
    """
    This concrete strategy evaluates a model's performance using recall.

    Attributes:
        None.

    Methods:
        - evaluate: This method calculates and returns the recall score based on true and predicted labels.
    """
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Evaluates a model's performance using recall.
    
        Args:
            y_true (pd.Series): true labels.
            y_pred (np.ndarray): predicted labels.
    
        Returns:
            recall score (float): indicating the proportion of true positive predictions relative to actual positives.
    """
        try:
            return recall_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error in recall evaluation: {e}")
            raise e

# Concrete strategy for F2 score
class F1ScoreEvaluationStrategy(ModelEvaluationStrategy):
    """
    This concrete strategy evaluates a model's performance using F1 score.

    Attributes:
        None.

    Methods:
        - evaluate: This method calculates and returns the F1 score based on true and predicted labels.
    """
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Evaluates a model's performance using F1 score.
    
        Args:
            y_true (pd.Series): true labels.
            y_pred (np.ndarray): predicted labels.
    
        Returns:
            F1 score (float): a harmonic mean of precision and recall, indicating the overall model performance.
        """
        try:
            return f1_score(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error in F{self.beta} score evaluation: {e}")
            raise e

# Concrete strategy for confusion matrix
class ConfusionMatrixEvaluationStrategy(ModelEvaluationStrategy):
    """
    This concrete strategy evaluates a model's performance using a confusion matrix.

    Attributes:
        None.

    Methods:
        - evaluate: This method calculates and returns the confusion matrix based on true and predicted labels.
    """
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Evaluates a model's performance using a confusion matrix.
    
        Args:
            y_true (pd.Series): true labels.
            y_pred (np.ndarray): predicted labels.
    
        Returns:
            confusion matrix (np.ndarray): a table indicating true positive, true negative, false positive, and false negative counts.
        """
        try:
            return confusion_matrix(y_true, y_pred)
        except Exception as e:
            logging.error(f"Error in confusion matrix evaluation: {e}")
            raise e

# ModelEvaluation class using the Strategy pattern
class ModelEvaluation:
    """
    This class uses the Strategy pattern to evaluate a machine learning model's performance based on a selected strategy.

    Attributes:
        - strategy (ModelEvaluationStrategy): The model evaluation strategy to be used.

    Methods:
        - __init__: Constructor that initializes the ModelEvaluation with a specific model evaluation strategy.

        - evaluate_model: This method evaluates a machine learning model's performance using the selected strategy and the true and predicted labels.
    
    """
    def __init__(self, strategy: ModelEvaluationStrategy):
        """
        Constructor that initializes the ModelEvaluation with a specific model evaluation strategy.
                
        Arguments:
            strategy (ModelEvaluationStrategy): The model evaluayion strategy to be used.
        Returns: 
            None.
        """
        self.strategy = strategy

    def evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray) -> Union[float, np.ndarray]:
        """Evaluates the model by using the provided evaluation strategy
        
        Arguments:
            y_true (pd.Series): true labels.
            y_pred (np.ndarray): predicted labels

        Returns:
            float (score) or a numpy array (confusion matrix).
        """
        return self.strategy.evaluate(y_true, y_pred)
