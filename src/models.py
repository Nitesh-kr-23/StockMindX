import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_forecaster(n_features: int, seq_len: int, n_horizons: int = 3,
                           hidden_size: int = 64, n_layers: int = 2,
                           dropout: float = 0.2) -> tf.keras.Model:
    """Stacked LSTM encoder + dense forecasting head."""
    inputs = layers.Input(shape=(seq_len, n_features), name="window")
    x = inputs
    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        x = layers.LSTM(hidden_size, return_sequences=return_sequences,
                         dropout=dropout, name=f"lstm_{i}")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_horizons, name="forecast")(x)
    return models.Model(inputs, outputs, name="LSTMForecaster")


def build_gru_forecaster(n_features: int, seq_len: int, n_horizons: int = 3,
                          hidden_size: int = 64, n_layers: int = 2,
                          dropout: float = 0.2) -> tf.keras.Model:
    """Stacked GRU encoder + dense forecasting head. GRUs train faster than
    LSTMs with similar accuracy on short sequences like ours."""
    inputs = layers.Input(shape=(seq_len, n_features), name="window")
    x = inputs
    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        x = layers.GRU(hidden_size, return_sequences=return_sequences,
                        dropout=dropout, name=f"gru_{i}")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_horizons, name="forecast")(x)
    return models.Model(inputs, outputs, name="GRUForecaster")


@keras.saving.register_keras_serializable(package="neuralhorizon")
class PositionalEncoding(layers.Layer):

    def __init__(self, d_model: int, max_len: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((max_len, d_model), dtype="float32")
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pe = tf.constant(pe[None, ...])

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len]

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config


def build_transformer_forecaster(n_features: int, seq_len: int, n_horizons: int = 3,
                                  d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                                  dropout: float = 0.2) -> tf.keras.Model:

    inputs = layers.Input(shape=(seq_len, n_features), name="window")
    x = layers.Dense(d_model, name="input_proj")(inputs)
    x = PositionalEncoding(d_model, name="pos_encoding")(x)

    for i in range(n_layers):
        attn_out = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout,
            name=f"self_attn_{i}",
        )(x, x)
        x = layers.LayerNormalization(name=f"ln1_{i}")(x + attn_out)
        ff = layers.Dense(d_model * 4, activation="relu", name=f"ff1_{i}")(x)
        ff = layers.Dense(d_model, name=f"ff2_{i}")(ff)
        ff = layers.Dropout(dropout)(ff)
        x = layers.LayerNormalization(name=f"ln2_{i}")(x + ff)

    pooled = layers.GlobalAveragePooling1D(name="pool")(x)
    pooled = layers.Dropout(dropout)(pooled)
    outputs = layers.Dense(n_horizons, name="forecast")(pooled)
    return models.Model(inputs, outputs, name="TransformerForecaster")


MODEL_BUILDERS = {
    "LSTM": build_lstm_forecaster,
    "GRU": build_gru_forecaster,
    "Transformer": build_transformer_forecaster,
}
