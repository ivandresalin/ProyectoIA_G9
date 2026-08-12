import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)


class BaselineEvaluator:

  def __init__(self):
    # Se añade class_weight='balanced' en los modelos que lo soportan
    self.modelos = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",  # 👈 Balanceo de clases
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",  # 👈 Balanceo de clases
            random_state=42,
        ),
        "Regresión Logística": LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            class_weight="balanced",  # 👈 Balanceo de clases
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }
    self.modelos_entrenados = {}

  def entrenar_y_evaluar_todos(
      self,
      X_train: np.ndarray,
      y_train: np.ndarray,
      X_test: np.ndarray,
      y_test: np.ndarray,
  ) -> pd.DataFrame:
    resultados = []

    for nombre, modelo in self.modelos.items():
      modelo.fit(X_train, y_train)
      self.modelos_entrenados[nombre] = modelo

      y_pred = modelo.predict(X_test)

      acc = accuracy_score(y_test, y_pred)
      prec = precision_score(
          y_test, y_pred, average="macro", zero_division=0
      )
      rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
      f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

      resultados.append({
          "Modelo": nombre,
          "Accuracy": round(acc, 4),
          "Precision (Macro)": round(prec, 4),
          "Recall (Macro)": round(rec, 4),
          "F1-Score (Macro)": round(f1, 4),
      })

    return pd.DataFrame(resultados)