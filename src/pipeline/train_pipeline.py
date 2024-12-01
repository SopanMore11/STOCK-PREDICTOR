from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.models.lstm_model import LSTMModel
from src.training.train import ModelTrainer
from src.utils import load_config
from src.logger import logging
from src.exception import CustomException
import numpy as np
import sys


# try:
data_ingestion = DataIngestion()
data_transformation = DataTransformation()

df_path = data_ingestion.initiate_data_ingestion()
data_transformation.initiate_data_transformation(data_path=df_path)

X_train = np.load('artifacts/X_train.npy', allow_pickle=True)
Y_train = np.load('artifacts/Y_train.npy', allow_pickle=True)
X_val = np.load('artifacts/X_val.npy', allow_pickle=True)
Y_val = np.load('artifacts/Y_val.npy', allow_pickle=True)

input_shape = (X_train.shape[1], X_train.shape[2])
lstm_model = LSTMModel(input_shape=input_shape)

config_path = "src/training/config.yaml"
model = lstm_model._build_model()
model_trainer = ModelTrainer(X_train, Y_train, X_val, Y_val, config_path=config_path)

history, model = model_trainer.train_model(model=model)
logging.info("Model Training is Completed.")

model_saving_path = "trained models/model_version_1.h5"
model.save(model_saving_path)
logging.info(f"Model is successfully saved at {model_saving_path}")
# except Exception as e:
#     raise CustomException(e, sys)