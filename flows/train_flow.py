from tasks.preprocess_data import data_preprocess
from tasks.model_train import train_model
from tasks.evaluate_model import evaluate_model
from tasks.ingest_data import ingest_df
from tasks.save_model import save_model
from tasks.experiment_track import track_experiment
from prefect import flow


@flow()
def workflow(data_path: str):
   """
   This Prefect workflow defines the end-to-end machine learning pipeline, including data ingestion, preprocessing,
   model training, evaluation, model saving, and experiment tracking.
   
   Args:
      data_path (str): The file path to the dataset for ingestion.
   
   Returns:
      None
   
   Raises:
      None.
   """
   df = ingest_df(data_path=data_path)
   X_train, X_test, y_train, y_test = data_preprocess(df)
   model = train_model(X_train=X_train, y_train=y_train)
   save_model(model=model)
   accuracy, precision, recall = evaluate_model(model, X_test, y_test)
   track_experiment(model, accuracy, precision, recall)