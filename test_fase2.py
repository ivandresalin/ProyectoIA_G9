import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from src.baseline_evaluator import BaselineEvaluator
from src.data_preprocessing import DataPreprocessor
from src.mlp_classifier import MLPClassifier

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("🚀 Iniciando prueba de la Fase 2 (Entrenamiento de Modelos)...")

# 1. Cargar y procesar datos con el pipeline de la Fase 1
ruta_inicio = (
    "data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx"
)
ruta_fin = "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.xlsx"

preprocesador = DataPreprocessor(ruta_inicio, ruta_fin)
df_raw = preprocesador.cargar_y_fusionar_datasets(sample_size=3000)
df_clean = preprocesador.limpiar_y_calcular_abandono(df_raw)
df_final = preprocesador.discretizar_riesgo(df_clean)
X, y = preprocesador.transformar_caracteristicas(df_final, is_training=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 2. Entrenar Modelo Propio (MLP en Keras)
print("\n🧠 Entrenando Modelo Propio (Perceptrón Multicapa - Keras)...")
mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=3)
mlp.entrenar(X_train, y_train, epochs=40, batch_size=32)

y_pred_mlp = mlp.predecir(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)

print(f"   └─ MLP Accuracy: {acc_mlp:.4f} | F1-Score Macro: {f1_mlp:.4f}")

# 3. Entrenar y Comparar los 5 Modelos de Línea Base (scikit-learn)
print("\n📊 Entrenando y evaluando 5 Modelos de Línea Base (scikit-learn)...")
evaluador = BaselineEvaluator()
df_comparativa = evaluador.entrenar_y_evaluar_todos(
    X_train, y_train, X_test, y_test
)

# Consolidar fila del MLP en la tabla comparativa
fila_mlp = pd.DataFrame([{
    "Modelo": "MLP (Propio - Keras)",
    "Accuracy": round(acc_mlp, 4),
    "Precision (Macro)": round(
        precision_score(
            y_test, y_pred_mlp, average="macro", zero_division=0
        ),
        4,
    ),
    "Recall (Macro)": round(
        recall_score(y_test, y_pred_mlp, average="macro", zero_division=0), 4
    ),
    "F1-Score (Macro)": round(f1_mlp, 4),
}])

tabla_final = pd.concat([fila_mlp, df_comparativa], ignore_index=True)

print("\n" + "=" * 65)
print("🏆 TABLA COMPARATIVA DE DESEMPEÑO (6 MODELOS EVALUADOS)")
print("=" * 65)
print(tabla_final.to_string(index=False))