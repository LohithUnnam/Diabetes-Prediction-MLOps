import pandas as pd
import logging
from prefect import task



class Ingest_data:
      """
      Ingesting the data from the data_path.
      """
      def __init__(self, data_path: str) -> None:
            """
            Args:
              data_path: path to the data
            """
            self.data_path = data_path

      def get_data(self):
            """
            Ingesting data from the data_path.
            """
            logging.info(f"Ingesting Data From {self.data_path}")
            return pd.read_csv(self.data_path)


@task
def ingest_df(data_path: str) -> pd.DataFrame:
      """
      This function ingests the data from the speicified data path.

      Args:
        data_path: path to the data
      
      Returns:
        pd.DataFrame: This function returns a dataframe of the csv file
      """

      try:
        ingest_data = Ingest_data(data_path)
        df = ingest_data.get_data()
        return df
      except Exception as e:
           logging.error(f"Error while ingesting the data")
           raise e