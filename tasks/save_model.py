from .config import ModelNameConfig
from prefect import task
import joblib
from sklearn.base import ClassifierMixin
import logging

@task
def save_model(model: ClassifierMixin):
    """
    This function saves a trained machine learning model to a file using joblib, based on the specified model name.

    Args:
        model (ClassifierMixin): The trained machine learning model to be saved.

    Returns:
        None

    Raises:
        ValueError: If the model name specified in `ModelNameConfig` is not recognized.
        Any other exceptions encountered during model saving.
    """
    
    try:
        model_name = ModelNameConfig.model_name
        model_paths = {
            "randomforest_classifier": 'C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\rfc_model.pkl',
            "logistic_regression": 'C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\lgr_model.pkl',
            "knn_classifier": 'C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\knc_model.pkl',
            "decisiontree_classifier": 'C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\dtc_model.pkl',
            "svm_classifier": 'C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\svc_model.pkl',
            "xgboost": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\xgb.pkl",
            "adaboost": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\adab.pkl"
        }

        if model_name in model_paths:
            joblib.dump(model, model_paths[model_name])
        else:
            logging.error(f"Unknown model name: {model_name}")
            raise ValueError(f"Unknown model name: {model_name}")

    except Exception as e:
        logging.error(f"Error while saving the model: {e}")
        raise e

    