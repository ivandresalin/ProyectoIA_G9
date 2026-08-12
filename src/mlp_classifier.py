import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class MLPClassifier:

  def __init__(
      self, input_dim: int, num_classes: int = 3, learning_rate: float = 0.001
  ):
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.learning_rate = learning_rate
    self.model = self._construir_arquitectura()

  def _construir_arquitectura(self) -> Sequential:
    """Diseña la arquitectura del Perceptrón Multicapa según especificaciones del proyecto."""
    model = Sequential([
        Input(shape=(self.input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(self.num_classes, activation="softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

  def entrenar(
      self,
      X_train: np.ndarray,
      y_train: np.ndarray,
      epochs: int = 50,
      batch_size: int = 32,
  ):
    y_train_cat = to_categorical(y_train, num_classes=self.num_classes)

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )

    history = self.model.fit(
        X_train,
        y_train_cat,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )
    return history

  def predecir(self, X: np.ndarray) -> np.ndarray:
    probabilidades = self.model.predict(X, verbose=0)
    return np.argmax(probabilidades, axis=1)