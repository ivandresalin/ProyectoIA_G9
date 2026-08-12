import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataPreprocessor
from src.mlp_classifier import MLPClassifier
from src.shap_explainer import SHAPExplainer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("🚀 Probando el Módulo de Explicabilidad (SHAP)...")

# 1. Cargar datos de prueba
ruta_inicio = (
    "data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx"
)
ruta_fin = "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.xlsx"

preprocesador = DataPreprocessor(ruta_inicio, ruta_fin)
df_raw = preprocesador.cargar_y_fusionar_datasets(sample_size=1500)
df_clean = preprocesador.limpiar_y_calcular_abandono(df_raw)
df_final = preprocesador.discretizar_riesgo(df_clean)
X, y = preprocesador.transformar_caracteristicas(df_final, is_training=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Entrenar MLP rápido
print("🧠 Entrenando MLP...")
mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=3)
mlp.entrenar(X_train, y_train, epochs=15, batch_size=32)

# 3. Calcular Explicabilidad SHAP
print("🔍 Calculando Shapley Values con SHAP...")
explainer = SHAPExplainer(
    mlp.model.predict, X_train, preprocesador.feature_names
)
shap_vals, muestra = explainer.calcular_explicabilidad(X_test, n_samples=5)

print("\n" + "=" * 55)
print("✅ ¡EXPLICABILIDAD CON SHAP CALCULADA CON ÉXITO!")
print("=" * 55)
print(f"📐 Tamaño de la muestra procesada por SHAP: {muestra.shape}")
print("Variables principales analizadas:", preprocesador.feature_names[:5])