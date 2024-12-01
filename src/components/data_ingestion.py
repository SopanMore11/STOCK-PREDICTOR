import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler


@dataclass
class DataIngestionConfig:
    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def read_csv_files_in_folder(self, folder_path):
        """
        Reads all CSV files in the specified folder into DataFrames.

        Parameters:
        - folder_path (str): The path to the folder containing the CSV files.

        Returns:
        - dict: A dictionary where keys are file names (without extensions) 
                and values are the corresponding DataFrames.
        """
        dataframes = []
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return dataframes

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):  # Check if the file is a CSV
                file_path = os.path.join(folder_path, file_name)
                
                try:
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                except Exception as e:
                    CustomException(e, sys)
        all_stocks_df = pd.concat(dataframes, ignore_index=True)
        return all_stocks_df

    
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:

            folder_path = self.ingestion_config.data_path
            df = self.read_csv_files_in_folder(folder_path)
            logging.info("Read the dataset as dataframes")

            # Saving the Complete data as the raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Data Ingestion is Completed.")

            return self.ingestion_config.raw_data_path
        except Exception as e:
            CustomException(e, sys)

    
            

# if __name__ == "__main__":
#     data_ingestion = DataIngestion()
#     data_ingestion.initiate_data_ingestion()
    # X_train, Y_train, X_val, Y_val, X_test, Y_test = data_ingestion.initiate_data_ingestion()
    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"X_val shape: {X_val.shape}")

    # print(f"y_train shape: {Y_train.shape}")
    # print(f"y_test shape: {Y_test.shape}")
    # print(f"y_val shape: {Y_val.shape}")
