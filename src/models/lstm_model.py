import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

class LSTMModel:
    def __init__(self, input_shape, lstm_units=50, dense_units=32, learning_rate=0.001, dropout_rates=(0.3, 0.3, 0.2), l2_reg=0.01):
        """
        Initializes the OptimizedLSTMModel with the specified parameters.

        Args:
            input_shape (tuple): Shape of the input data (timesteps, features).
            lstm_units (int): Number of units in each LSTM layer.
            dense_units (int): Number of units in the Dense layer.
            learning_rate (float): Learning rate for the Adam optimizer.
            dropout_rates (tuple): Dropout rates for the LSTM and Dense layers.
            l2_reg (float): L2 regularization parameter for kernel regularization.
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.dropout_rates = dropout_rates
        self.l2_reg = l2_reg

    def _build_model(self):
        """
        Builds the optimized LSTM model architecture.

        Returns:
            model (tf.keras.Model): Compiled LSTM model.
        """
        model = Sequential()

        # First Bidirectional LSTM layer
        model.add(
            Bidirectional(
                LSTM(self.lstm_units, return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
                input_shape=self.input_shape
            )
        )
        model.add(Dropout(self.dropout_rates[0]))

        # Second LSTM layer
        model.add(LSTM(self.lstm_units, return_sequences=True))
        model.add(Dropout(self.dropout_rates[1]))

        # Third LSTM layer
        model.add(LSTM(self.lstm_units))
        model.add(Dropout(self.dropout_rates[1]))

        # Dense layers
        model.add(Dense(self.dense_units, activation="relu"))
        model.add(Dropout(self.dropout_rates[2]))

        # Output layer
        model.add(Dense(1))  # Regression output for stock price prediction

        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss="mean_squared_error")

        return model

if __name__ == "__main__":
    model = LSTMModel(input_shape=(60, 20))
    lstm_model = model._build_model()
    print(lstm_model.summary())
