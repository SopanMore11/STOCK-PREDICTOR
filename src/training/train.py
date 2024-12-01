from src.models.lstm_model import LSTMModel
from src.utils import load_config
from src.logger import logging
from src.exception import CustomException
import sys


class ModelTrainer:
    def __init__(self, X_train, Y_train, X_val, Y_val, config_path):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

        self.config = load_config(config_path=config_path)

    def train_model(self, model):
        try:
            logging.info("Started Model Training")
            history = model.fit(
                self.X_train, self.Y_train,
                validation_data = (self.X_val, self.Y_val),
                epochs = self.config["epochs"],
                batch_size = self.config["batch_size"]
            )

            return history, model
        except Exception as e:
            CustomException(e, sys)
