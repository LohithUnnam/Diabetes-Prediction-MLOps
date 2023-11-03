from abc import ABC, abstractmethod
import logging
import joblib
from typing import Union
from typing import Tuple
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessingStrategy(ABC):
    """
    This abstract base class defines a common interface for data processing strategies.

    Attributes:
        None.

    Methods:
        - process: It processes data and returns the result, which can be a DataFrame or Series.
    """
    @abstractmethod
    def process(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        This is an abstract method for data processing strategies.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing data to be processed.

        Returns:
            The processed data, which could be a DataFrame or Series.
        """
        pass

class ConvertAndDropDuplicatesStrategy(DataProcessingStrategy):
    """This concrete strategy is responsible for converting and dropping duplicates
    
    Attributes: 
        None.

    Methods:
       process: This method converts DataFrame columns to 'int' type and drops duplicates.
    """
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method converts each column in the input DataFrame to the 'int' data type and then removes duplicate rows from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input Pandas DataFrame to be processed.
        
        Returns:
            Processed Pandas DataFrame with columns converted to 'int' and duplicate rows removed.
        """
        try:
            for column in df.columns:
                df[column] = df[column].astype('int')
            df.drop_duplicates(inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error while converting dloat to int and dropping duplicate rows: {e}")
            raise e
        

class LoadDataStrategy(DataProcessingStrategy):
    """
    This concrete strategy for loading data from a DataFrame.        
    """
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        This concrete method loads and prepares the input data.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing data to be processed.

        Returns:
            pd.DataFrame: dataframe which contains features
            pd.Series: series which contain label
        """
        try:
            X = df.drop(columns=['Diabetes_binary'])
            y = df['Diabetes_binary']
            return X, y
        except Exception as e:
            logging.error(f"Error in loading data: {e}")
            raise e

class ColumnRemovalStrategy(DataProcessingStrategy):
    """
    FeatureSelectionStrategy is a concrete strategy for feature selection that removes specified columns from a DataFrame.
    """
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This method removes the specified columns ('Income' and 'Smoker') from the input DataFrame.

        Args:
            df (pd.DataFrame): Input Pandas DataFrame containing the dataset.
        
        Returns:
            Pandas DataFrame with the specified columns removed, representing the selected features.
        """
        try:
            # Remove specified columns
            columns_to_remove = []
            df = df.drop(columns=columns_to_remove)
            return df
        except Exception as e:
            logging.error(f"Error while removing unnecessary columns: {e}")
            raise e
        

class ApplySmoteStrategy(DataProcessingStrategy):
    """
    This concrete strategy applies the Synthetic Minority Over-sampling Technique (SMOTE) to balance class distribution.

    Attributes:
        None.

    Methods:
        - process: This method takes feature data and labels, applies SMOTE, and returns resampled data as a tuple.
    """
    def process(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        This concrete method applies SMOTE to balance the class distribution.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Label Series.

        Returns:
            pd.DataFrame: Resampled feature DataFrame
            pd.Series: label Series.
        """
        try:
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            logging.error(f"Error in applying SMOTE: {e}")
            raise e


class ScaleFeaturesStrategy(DataProcessingStrategy):
    """
    This concrete strategy scales specified columns using StandardScaler and saves the scaler model.

    Attributes:
        None.

    Methods:
        - process: This method scales specified columns in training and testing data and returns the scaled data.
    """



    def process(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This concrete method scales specified columns using StandardScaler and saves the scaler model.
        
        Args:
        X_train (pd.DataFrame): Training feature DataFrame.
        X_test (pd.DataFrame): Testing feature DataFrame.
        
        Returns:
           pd.DataFrame: Scaled training feature DataFrame
           pd.DataFrame: Scaled testing feature DataFrame.
        """
        try:
            columns_to_scale = ['BMI','MentHlth','PhysHlth']
            scaler = StandardScaler()
            X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
            X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
            joblib.dump(scaler, 'C:\\Users\\HP\\Desktop\\diabetes_predictor\\saved_model\\scaler_model.pkl')
            return X_train, X_test
        
        except Exception as e:
            logging.error(f"Error in scaling features: {e}")
            raise e

class SplitDataStrategy(DataProcessingStrategy):
    """
    This concrete strategy splits data into training and testing sets.

    Attributes:
        None.

    Methods:
        - process: This method performs a train-test split on resampled data and returns training and testing data.
    """
    def process(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        This concrete method splits the data into training and testing sets.

        Args:
            X (pd.DataFrame):  feature DataFrame.
            y (pd.Series):  label Series.

        Returns:
            pd.DataFrame: Training features DataFrame
            pd.DataFrame: Testing features DataFrame
            pd.Series: Training label Series
            pd.Series: Testing label Series
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in splitting data: {e}")
            raise e

class DataPreprocessor:
    """
    This class uses the Strategy pattern to process data based on the selected strategy.

    Attributes:
        - strategy (DataProcessingStrategy): The data processing strategy to be used.

    Methods:
        - __init__: Constructor that initializes the DataPreprocessor with a specific data processing strategy.

        - process_data: This method takes any number of arguments and it uses a specific strategy to handle and work with the data using those inputs.
          It returns the processed data based on the chosen strategy.
         
    """
    def __init__(self, strategy: DataProcessingStrategy):
        """
        This method initializes the DataPreProcessor with specified strategy.

        Args:
            strategy (DataProcessingStrategy): The data processing strategy to be used.
        """
        self.strategy = strategy

    def process_data(self, *args):
        """
        This method Processes data using the selected strategy.

        Args:
            *args: Any number of arguments to be passed to the strategy's 'process' method.

        Returns:
            The processed data as specified by the strategy.
        """
        return self.strategy.process(*args)

