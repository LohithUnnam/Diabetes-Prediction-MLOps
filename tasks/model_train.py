from src.model_dev import (RandomForestClassifierStrategy, ModelDevelopment, LogisticRegressionStrategy, KNeighborsClassifierStrategy, DecisionTreeClassifierStrategy, 
SVMClassifierStrategy, AdaBoostStrategy, XGBoostStrategy)
from .config import ModelNameConfig
import logging
from sklearn.base import ClassifierMixin
from prefect import task
import pandas as pd

@task
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
   """
    This function trains a machine learning model based on the specified model name in the `ModelNameConfig`.

    Args:
        X_train (pd.DataFrame): The feature DataFrame for training.
        y_train (pd.Series): The label Series for training.

    Returns:
        ClassifierMixin: The trained machine learning model based on the selected model name.
        
    Raises:
        ValueError: If the model name specified in `ModelNameConfig` is not recognized.
    """
   try:
      if ModelNameConfig.model_name == "randomforest_classifier":
         # Deafult hyperparameters -> n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
         # max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, 
         # warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None
         rfc_strategy = RandomForestClassifierStrategy()
         model_development = ModelDevelopment(rfc_strategy)
         model = model_development.model_train(X_train, y_train, max_depth=100)
         return model
      
      elif ModelNameConfig.model_name == "xgboost":
         xgboost_strategy = XGBoostStrategy()
         model_development = ModelDevelopment(xgboost_strategy)
         model = model_development.model_train(X_train, y_train)
         return model
      elif ModelNameConfig.model_name == "adaboost":
         adaboost_strategy = AdaBoostStrategy()
         model_development = ModelDevelopment(adaboost_strategy)
         model = model_development.model_train(X_train, y_train)
         return model
      elif ModelNameConfig.model_name == "logistic_regression":
         lgr_strategy = LogisticRegressionStrategy()
         model_development = ModelDevelopment(lgr_strategy)
         model = model_development.model_train(X_train, y_train)
         return model
      
      elif ModelNameConfig.model_name == "knn_classifier":
         knnc_strategy = KNeighborsClassifierStrategy()
         model_development = ModelDevelopment(knnc_strategy)
         model = model_development.model_train(X_train, y_train)
         return model
      
      elif ModelNameConfig.model_name == "decisiontree_classifier":
         dtc_strategy = DecisionTreeClassifierStrategy()
         model_development = ModelDevelopment(dtc_strategy)
         model = model_development.model_train(X_train, y_train)
         return model
      
      elif ModelNameConfig.model_name == "svm_classifier":
         svmc_strategy = SVMClassifierStrategy()
         model_development = ModelDevelopment(svmc_strategy)
         model = model_development.model_train(X_train, y_train)
         return model
      
      else: 
         logging.error(f"Unknown model name: {ModelNameConfig.model_name}")
         raise ValueError(f"Unknown model name: {ModelNameConfig.model_name}")
   except Exception as e:
      logging.error(f"Erro while training model: {e}")
      raise e
