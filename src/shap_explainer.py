import matplotlib.pyplot as plt
import numpy as np
import shap


class SHAPExplainer:

  def __init__(
      self, model_predict_fn, X_train_sample: np.ndarray, feature_names: list
  ):
    """Inicializa el explicador SHAP usando KernelExplainer.

    :param model_predict_fn: Función de predicción del modelo (ej:
    mlp.model.predict)
    :param X_train_sample: Muestra del conjunto de entrenamiento para la línea
    base
    :param feature_names: Nombres de las variables/características procesadas
    """
    self.predict_fn = model_predict_fn
    self.feature_names = feature_names

    # Usamos una muestra de fondo de máximo 30 instancias para acelerar el cálculo
    num_samples = min(30, len(X_train_sample))
    self.background = shap.sample(X_train_sample, num_samples)
    self.explainer = shap.KernelExplainer(self.predict_fn, self.background)

  def calcular_explicabilidad(
      self, X_sample: np.ndarray, n_samples: int = 10
  ):
    """Calcula Shapley values sobre una muestra de prueba."""
    muestra = X_sample[: min(n_samples, len(X_sample))]
    shap_values = self.explainer.shap_values(muestra, nsamples=100)
    return shap_values, muestra

  def generar_grafico_resumen(self, X_sample: np.ndarray, n_samples: int = 10):
    """Genera una figura de matplotlib con el gráfico Summary de SHAP."""
    shap_values, muestra = self.calcular_explicabilidad(X_sample, n_samples)

    fig, ax = plt.subplots(figsize=(10, 6))

    if isinstance(shap_values, list):
      # Tomamos la clase 2 (Alto Riesgo) si es lista multiclase
      idx = min(2, len(shap_values) - 1)
      shap_vals_plot = shap_values[idx]
    else:
      shap_vals_plot = shap_values

    shap.summary_plot(
        shap_vals_plot,
        muestra,
        feature_names=self.feature_names,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    return fig