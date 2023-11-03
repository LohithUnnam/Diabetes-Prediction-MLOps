import mlflow
from sklearn.base import ClassifierMixin
from prefect import task
from .config import ModelNameConfig
import logging

@task
def track_experiment(model: ClassifierMixin, accuracy: float, precision: float, recall: float):
    """
    This function tracks an experiment using MLflow by logging hyperparameters, metrics, and artifacts.

    Args:
        model (ClassifierMixin): The trained machine learning model to track.
        accuracy (float): The accuracy metric to log.
        precision (float): The precision metric to log.
        recall (float): The recall metric to log.

    Returns:
        None

    Raises:
        ValueError: If the model name specified in `ModelNameConfig` is not recognized.
        Any other exceptions encountered during tracking.
    """
    
    try:
        active_run = mlflow.active_run()
        artifact_paths = {
        "randomforest_classifier": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\rfc_model.pkl",
        "logistic_regression": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\lgr_model.pkl",
        "knn_classifier": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\knc_model.pkl",
        "decisiontree_classifier": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\dtc_model.pkl",
        "svm_classifier": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\svc_model.pkl",
        "xgboost": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\xgb.pkl",
        "adaboost": "C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\adab.pkl"
        }
        if active_run is not None:
            hyper_parameters = model.get_params()
            
            for key, value in hyper_parameters.items():
                mlflow.log_param(key, value)
            
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            
            model_name = ModelNameConfig.model_name
            
            if model_name in artifact_paths:
                mlflow.log_artifact(artifact_paths[model_name])
            else:
                logging.error(f"Unknown model name: {model_name}")
                raise ValueError(f"Unknown model name: {model_name}")
        else:
            mlflow.start_run()
            hyper_parameters = model.get_params()
            
            for key, value in hyper_parameters.items():
                mlflow.log_param(key, value)
                        
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
                        
            model_name = ModelNameConfig.model_name
                        
            if model_name in artifact_paths:
                mlflow.log_artifact(artifact_paths[model_name])
                mlflow.end_run()
            else:
                logging.error(f"Unknown model name: {model_name}")
                raise ValueError(f"Unknown model name: {model_name}")

    except Exception as e:
        logging.error(f"Error in tracking experiment: {str(e)}")
        raise e  


  
