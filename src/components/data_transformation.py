import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
# from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_ingestion import DataIngestion, DataIngestionConfig

@dataclass
class DataTransformationConfig:
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    X_train_data_path: str = os.path.join('artifacts', 'X_train.npy')
    Y_train_data_path: str = os.path.join('artifacts', 'Y_train.npy')
    X_test_data_path: str = os.path.join('artifacts', 'X_test.npy')
    Y_test_data_path: str = os.path.join('artifacts', 'Y_test.npy')
    X_val_data_path: str = os.path.join('artifacts', 'X_val.npy')
    Y_val_data_path: str = os.path.join('artifacts', 'Y_val.npy')



class DataTransformation:
    def __init__(self) -> None:
        self.trasnformation_config = DataTransformationConfig()
    
    def encode_company_name(self, df, company_column='Company_name'):
        """
        One-hot encodes the 'Company_name' column in the dataframe.

        Parameters:
            df (pd.DataFrame): The input dataframe with a 'Company_name' column.
            company_column (str): The name of the column containing company names.

        Returns:
            pd.DataFrame: A new dataframe with one-hot encoded company names.
        """
        # Get one-hot encoded columns
        one_hot_encoded = pd.get_dummies(df[company_column], prefix="Company", dtype=int)
        
        # Drop the original company column and concatenate one-hot encoded columns
        df_encoded = pd.concat([df, one_hot_encoded], axis=1)
    
        return df_encoded

    def create_sequences(self, df, feature_columns, target_column, sequence_length=60):
        """
        Create sequences for LSTM training where the last `sequence_length` days are used 
        to predict the next day's target.

        Parameters:
        - df: DataFrame with features and target.
        - feature_columns: List of column names to be used as features.
        - target_column: The column name for the target (e.g., 'Close').
        - sequence_length: The number of past days to use for each sequence.

        Returns:
        - X: Numpy array of shape (num_sequences, sequence_length, num_features).
        - Y: Numpy array of shape (num_sequences, 1).
        """
        X, Y = [], []
        
        for company, company_df in df.groupby('Company_name'):
            data = company_df[feature_columns + [target_column]].values  # Select relevant columns
            for i in range(sequence_length, len(data)):
                X.append(data[i-sequence_length:i, :-1])  # All feature columns
                Y.append(data[i, -1])  # Target column
            
        return np.array(X), np.array(Y)

    def scale_dataframe(self, df, target_column, exclude_columns=None, feature_range=(0, 1)):
        """
        Scales the features and target column in a DataFrame using MinMaxScaler.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to be scaled.
        - target_column (str): Name of the target column to scale.
        - exclude_columns (list): List of column names to exclude from scaling. Defaults to None.
        - feature_range (tuple): The desired range of transformed data. Defaults to (0, 1).

        Returns:
        - scaled_df (pd.DataFrame): DataFrame with scaled features and target.
        - feature_scaler (MinMaxScaler): Fitted scaler for input features.
        - target_scaler (MinMaxScaler): Fitted scaler for the target column.
        """
        if exclude_columns is None:
            exclude_columns = []

        # Identify feature columns
        feature_columns = [col for col in df.columns if col not in exclude_columns + [target_column]]

        # Initialize scalers
        feature_scaler = MinMaxScaler(feature_range=feature_range)
        target_scaler = MinMaxScaler(feature_range=feature_range)

        # Fit and transform features and target
        scaled_features = feature_scaler.fit_transform(df[feature_columns])
        scaled_target = target_scaler.fit_transform(df[[target_column]])

        # Create a new DataFrame for scaled data
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)
        scaled_df[target_column] = scaled_target

        # Add excluded columns back to the scaled DataFrame
        for col in exclude_columns:
            if col in df.columns:
                scaled_df[col] = df[col].values

        return scaled_df, feature_scaler, target_scaler
    
    def split_data_by_company(self, X, Y, company_names, train_ratio=0.7, val_ratio=0.15):
        """
        Split data into training, validation, and testing datasets by company.

        Parameters:
        - X: Numpy array of input features (num_samples, sequence_length, num_features).
        - Y: Numpy array of target values (num_samples, 1).
        - company_names: List of company names corresponding to each sample in X and Y.
        - train_ratio: Proportion of data to use for training.
        - val_ratio: Proportion of data to use for validation.

        Returns:
        - X_train, Y_train: Training data.
        - X_val, Y_val: Validation data.
        - X_test, Y_test: Testing data.
        """
        X_train, Y_train, X_val, Y_val, X_test, Y_test = [], [], [], [], [], []

        unique_companies = np.unique(company_names)

        for company in unique_companies:
            company_mask = (company_names == company)  # Select rows for this company
            X_company = X[company_mask]
            Y_company = Y[company_mask]
            
            # Determine split indices
            n_samples = len(X_company)
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)

            # Split the data
            X_train.append(X_company[:train_end])
            Y_train.append(Y_company[:train_end])
            X_val.append(X_company[train_end:val_end])
            Y_val.append(Y_company[train_end:val_end])
            X_test.append(X_company[val_end:])
            Y_test.append(Y_company[val_end:])

        # Concatenate splits across all companies
        X_train = np.concatenate(X_train)
        Y_train = np.concatenate(Y_train)
        X_val = np.concatenate(X_val)
        Y_val = np.concatenate(Y_val)
        X_test = np.concatenate(X_test)
        Y_test = np.concatenate(Y_test)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def initiate_data_transformation(self, data_path):

        logging.info("Initiated Data Transformation.")
        df = pd.read_csv(data_path)
        # Encoding the DataFrame
        df_encoded = self.encode_company_name(df=df)
        logging.info("Encodede the DataFrame Successfully")

        # Scaling the DataFrame
        scaled_df, feature_scaler, target_scaler = self.scale_dataframe(df_encoded, 'Close', ['Company_name'])
        cleaned_df = scaled_df.groupby("Company_name").apply(lambda group: group.iloc[60:])

        # Drop the extra index added by `groupby`
        cleaned_df.reset_index(drop=True, inplace=True)
        logging.info("DataFrame is Successfully Scaled.")

        # Create Lagging Features
        feature_columns = [col for col in scaled_df.columns if col not in ['Company_name']]
        X, Y = self.create_sequences(scaled_df, feature_columns, 'Close')
        logging.info("Created Lagging Features")

        # Splitting Data into train, Test and Validation
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.split_data_by_company(
            X, Y, cleaned_df['Company_name'].values
        )

        logging.info("Data Splitting is Done Successfully")
       
        os.makedirs(os.path.dirname(self.trasnformation_config.X_train_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.trasnformation_config.Y_train_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.trasnformation_config.X_test_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.trasnformation_config.Y_test_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.trasnformation_config.X_val_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.trasnformation_config.Y_val_data_path), exist_ok=True)

        np.save(self.trasnformation_config.X_train_data_path, X_train)
        np.save(self.trasnformation_config.Y_train_data_path, Y_train)
        np.save(self.trasnformation_config.X_val_data_path, X_val)
        np.save(self.trasnformation_config.Y_val_data_path, Y_val)
        np.save(self.trasnformation_config.X_test_data_path, X_test)
        np.save(self.trasnformation_config.Y_test_data_path, Y_test)



        logging.info("Data Transformation is Completed.")


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_transformation = DataTransformation()
    data_path = data_ingestion.initiate_data_ingestion()
    data_transformation.initiate_data_transformation(data_path=data_path)
    print("Done!")