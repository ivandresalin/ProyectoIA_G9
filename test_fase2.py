import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from src.baseline_evaluator import BaselineEvaluator
from src.data_preprocessing import DataPreprocessor
from src.mlp_classifier import MLPClassifier

print("🚀 Ejecutando Fase 2 con División Temporal y Control de Leakage...")

ruta_inicio = (
    "data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx"
)
ruta_fin = "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.xlsx"

preprocesador = DataPreprocessor(ruta_inicio, ruta_fin)
df_raw = preprocesador.cargar_y_fusionar_datasets(sample_size=3500)
df_clean = preprocesador.limpiar_y_calcular_abandono(df_raw)
df_final = preprocesador.discretizar_riesgo(df_clean)

X, y = preprocesador.transformar_caracteristicas(df_final, is_training=True)

# Divisón cronológica (Time-based split)
X_train, X_test, y_train, y_test, info_split = preprocesador.dividir_por_tiempo(
    df_final, X, y
)

print(f"📅 Criterio de Partición Temporal: {info_split}")
print(f"📊 Registros Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"🛡️ Variables en X (Entrada): {preprocesador.feature_names[:6]} ...")

# 1. Entrenar MLP
print("\n🧠 Entrenando MLP (Modelo Propio)...")
mlp = MLPClassifier(input_dim=X_train.shape[1], num_classes=3)
mlp.entrenar(X_train, y_train, epochs=40, batch_size=32)

y_pred_mlp = mlp.predecir(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)

# 2. Entrenar Modelos Base con Class Weight Balanced
print("📊 Entrenando 5 Modelos de Línea Base (scikit-learn)...")
evaluador = BaselineEvaluator()
df_baseline = evaluador.entrenar_y_evaluar_todos(
    X_train, y_train, X_test, y_test
)

# 3. Consolidar Tabla Comparativa
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

tabla_final = pd.concat([fila_mlp, df_baseline], ignore_index=True)

print("\n" + "=" * 65)
print("🏆 TABLA COMPARATIVA CON PARTICIÓN TEMPORAL (6 MODELOS)")
print("=" * 65)
print(tabla_final.to_string(index=False))