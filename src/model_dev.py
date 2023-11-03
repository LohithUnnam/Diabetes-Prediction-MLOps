from abc import ABC, abstractmethod
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin


class ModelDevelopmentStrategy(ABC):
    """
    This abstract base class defines a common interface for model development strategies.

    Attributes:
        None.

    Methods:
        - develop_model: It develops a machine learning model using the provided training data and any additional keyword arguments.
    """
    @abstractmethod
    def develop_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """
        This abstract method must be implemented by concrete strategy classes. It is responsible for developing a machine learning model using the provided training data 
        and optional keyword arguments.

        Arguments:
            X_train (pd.DataFrame): The feature DataFrame for training.
            y_train (pd.Series): The label Series for training.
            **kwargs (dict, optional): Optional keyword arguments that can be used for hyperparameter configuration.
        
        Returns:
            The machine learning model developed based on the specific strategy.
        """
        pass

class AdaBoostStrategy(ModelDevelopmentStrategy):
    def develop_model(self, X_train, y_train):
        try:
            base_classifier = DecisionTreeClassifier(max_depth=100)
            model = AdaBoostClassifier(base_classifier)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error in AdaBoost model development: {e}")
            raise e

class XGBoostStrategy(ModelDevelopmentStrategy):
    def develop_model(self, X_train, y_train):
        try:
            model = XGBClassifier()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error in XGBoost model development: {e}")
            raise e
                
class LogisticRegressionStrategy(ModelDevelopmentStrategy):
    """
    This concrete strategy develops a Logistic Regression model.

    Attributes:
        None.

    Methods:
        - develop_model: This method creates and trains a Logistic Regression model using the provided training data.
    """
    def develop_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """This method creates and trains a Logistic Regression model using the provided training data.
        
        Arguments:
            X_train (pd.DataFrame): The feature DataFrame for training.
            y_train (pd.Series): The label Series for training.
            **kwargs (dict, optional): Optional keyword arguments for configuring the model's hyperparameters.
            
        Returns:
            The trained Logistic Regression model.
        """
        try:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error in Logistic Regression model development: {e}")
            raise e


class KNeighborsClassifierStrategy(ModelDevelopmentStrategy):
    """
    This concrete strategy develops a k-Nearest Neighbors (KNN) model.

    Attributes:
        None.

    Methods:
        - develop_model: This method creates and trains a KNN model using the provided training data.
    """
    def develop_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """Creates and trains a k-Nearest Neighbors (KNN) model using the provided training data.
        
        Arguments:
            X_train (pd.DataFrame): The feature DataFrame for training.
            y_train (pd.Series): The label Series for training.
            **kwargs (dict, optional): Optional keyword arguments for configuring the model's hyperparameters.
        
        Returns:
            The trained KNN model.
        """
        try:
            model = KNeighborsClassifier()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error in k-Nearest Neighbors model development: {e}")
            raise e


class DecisionTreeClassifierStrategy(ModelDevelopmentStrategy):
    """
    This concrete strategy develops a Decision Tree Classifier model.

    Attributes:
        None.

    Methods:
        - develop_model: This method creates and trains a Decision Tree Classifier model using the provided training data.
    """
    def develop_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """
        This method creates and trains a Decision Tree Classifier model using the provided training data.
        
        Arguments:
            X_train (pd.DataFrame): The feature DataFrame for training.
            y_train (pd.Series): The label Series for training.
            **kwargs (dict, optional): Optional keyword arguments for configuring the model's hyperparameters.
        
        Returns:
            The trained Decision Tree Classifier model.
        """
        try:
            model = DecisionTreeClassifier()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error in Decision Tree Classifier model development: {e}")
            raise e


class RandomForestClassifierStrategy(ModelDevelopmentStrategy):
    """
    This concrete strategy develops a Random Forest Classifier model.

    Attributes:
        None.

    Methods:
        - develop_model: This method creates and trains a Random Forest Classifier model using the provided training data and optional keyword arguments for hyperparameters.
    """
    def develop_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """
        This method creates and trains a Random Forest Classifier model using the provided training data and optional keyword arguments for hyperparameter configuration.
        
        Arguments:
            X_train (pd.DataFrame): The feature DataFrame for training.
            y_train (pd.Series): The label Series for training.
            **kwargs (dict, optional): Optional keyword arguments for configuring the model's hyperparameters.
        
        Returns:
            The trained Random Forest Classifier model.
        """
        try:
            n_estimators = kwargs.get('n_estimators', 100)  # Default value of n_estimators is 100
            max_depth = kwargs.get('max_depth', None)  # Default value of max_depth is None
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error in Random Forest Classifier model development: {e}")
            raise e


class SVMClassifierStrategy(ModelDevelopmentStrategy):
    """
    This concrete strategy develops a Support Vector Machine (SVM) Classifier model.

    Attributes:
        None.

    Methods:
        - develop_model: This method creates and trains an SVM Classifier model using the provided training data.
    """

    def develop_model(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """Creates and trains a Support Vector Machine (SVM) Classifier model using the provided training data.
        
        Arguments:
            X_train (pd.DataFrame): The feature DataFrame for training.
            y_train (pd.Series): The label Series for training.
            **kwargs (dict, optional): Optional keyword arguments for configuring the model's hyperparameters.
        
        Returns:
            The trained SVM Classifier model.
        """
        try:
            model = SVC()
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logging.error(f"Error in SVM Classifier model development: {e}")
            raise e


class ModelDevelopment:
    """
    This class uses the Strategy pattern to develop machine learning models based on the selected strategy.

    Attributes:
        - strategy (ModelDevelopmentStrategy): The model development strategy to be used.

    Methods:
        - __init__: Constructor that initializes the ModelDevelopment with a specific model development strategy.

        - develop_model: This method develops a machine learning model using the selected strategy and provided training data.
    """
    def __init__(self, strategy: ModelDevelopmentStrategy):
        """
        Constructor that initializes the ModelDevelopment with a specific model development strategy.
        
        Arguments:
            strategy (ModelDevelopmentStrategy): The model development strategy to be used.
        Returns: 
            None.
        """
        self.strategy = strategy

    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs) -> ClassifierMixin:
        """
        Develops a machine learning model using the selected strategy and provided training data.
        
        Arguments:
        X_train (pd.DataFrame): The feature DataFrame for training.
        y_train (pd.Series): The label Series for training.
        **kwargs (dict, optional): Optional keyword arguments for configuring the model's hyperparameters.
        
        Returns:
            The machine learning model developed based on the specific strategy.
        """
        return self.strategy.develop_model(X_train, y_train, **kwargs)

    

