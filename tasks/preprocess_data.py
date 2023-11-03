from src.data_cleaning import LoadDataStrategy, ScaleFeaturesStrategy, SplitDataStrategy, DataPreprocessor, ApplySmoteStrategy, ColumnRemovalStrategy, ConvertAndDropDuplicatesStrategy
from prefect import task
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple


@task
def data_preprocess(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    This function preprocesses the input DataFrame using a series of data processing strategies and saves the preprocessed data.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the input data to be processed.

    Returns:
        pd.DataFrame (X_train): Training features DataFrame
        pd.DataFrame (X_test): Testing features DataFrame
        pd.Series (y_train): Training label Series
        pd.Series (y_test)): Testing label Series
    """
    
    convert_drop_duplicate_strategy = ConvertAndDropDuplicatesStrategy()
    preprocessor = DataPreprocessor(convert_drop_duplicate_strategy)
    df = preprocessor.process_data(df)
    load_strategy = LoadDataStrategy()
    preprocessor.strategy = load_strategy
    X, y = preprocessor.process_data(df)
    split_data_strategy = SplitDataStrategy()
    preprocessor.strategy = split_data_strategy
    X_train, X_test, y_train, y_test = preprocessor.process_data(X, y)
    apply_smote_strategy = ApplySmoteStrategy()
    preprocessor.strategy = apply_smote_strategy
    X_train, y_train = preprocessor.process_data(X_train, y_train)
    scale_features_strategy = ScaleFeaturesStrategy()
    preprocessor.strategy = scale_features_strategy   
    X_train, X_test = preprocessor.process_data(X_train, X_test)
    # Combine X_train and X_test into a single DataFrame
    X = pd.concat([X_train, X_test])
    # Combine y_train and y_test into a single Series (if they are Series)
    y = pd.concat([y_train, y_test])
    combined_data = pd.concat([X, y], axis=1)
   # Save the combined DataFrame to a CSV file
    combined_data.to_csv("C:\\Users\\HP\\Desktop\\diabetes_predictor\\data\\diabetes_preprocessed.csv", index=False)
    return X_train, X_test, y_train, y_test
