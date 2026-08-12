import matplotlib.pyplot as plt
import numpy as np
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
from src.shap_explainer import SHAPExplainer
import streamlit as st

st.set_page_config(
    page_title="Sistema de Alerta de Deserción Escolar - MINEDUC",
    layout="wide",
    page_icon="🏫",
)

st.title("🏫 Sistema de Clasificación y Alerta Temprana de Deserción Escolar")
st.caption(
    "Proyecto Final de Inteligencia Artificial - ESPOL | Red Neuronal MLP,"
    " Comparativa con 5 Modelos e Interpretabilidad SHAP"
)

# Panel lateral de configuración
st.sidebar.header("⚙️ Configuración del Pipeline")
sample_size = st.sidebar.slider(
    "Tamaño de muestra de datos",
    min_value=1000,
    max_value=10000,
    value=3000,
    step=500,
)


@st.cache_data
def cargar_y_procesar_datos(sample_size):
  ruta_inicio = (
      "data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx"
  )
  ruta_fin = "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.xlsx"

  preprocesador = DataPreprocessor(ruta_inicio, ruta_fin)
  df_raw = preprocesador.cargar_y_fusionar_datasets(sample_size=sample_size)
  df_clean = preprocesador.limpiar_y_calcular_abandono(df_raw)
  df_final = preprocesador.discretizar_riesgo(df_clean)

  X, y = preprocesador.transformar_caracteristicas(df_final, is_training=True)
  return preprocesador, df_final, X, y


# Cargar datos con barra de progreso de Streamlit
with st.spinner("Cargando y procesando dataset histórico del MINEDUC..."):
  preprocesador, df_final, X, y = cargar_y_procesar_datos(sample_size)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


@st.cache_resource
def entrenar_modelos(_X_train, _y_train, _X_test, _y_test, input_dim):
  # Entrenamiento del Modelo Propio
  mlp = MLPClassifier(input_dim=input_dim, num_classes=3)
  mlp.entrenar(_X_train, _y_train, epochs=30, batch_size=32)

  # Entrenamiento de los 5 modelos de línea base
  evaluador = BaselineEvaluator()
  df_baseline = evaluador.entrenar_y_evaluar_todos(
      _X_train, _y_train, _X_test, _y_test
  )

  return mlp, evaluador, df_baseline


with st.spinner("Entrenando Perceptrón Multicapa (MLP) y modelos base..."):
  mlp, evaluador, df_baseline = entrenar_modelos(
      X_train, y_train, X_test, y_test, X_train.shape[1]
  )

# Pestañas principales de la interfaz
tab1, tab2, tab3 = st.tabs(
    ["🚦 Semáforo de Riesgo", "🏆 Comparativa de Modelos", "🔍 Explicabilidad (SHAP)"]
)

with tab1:
  st.subheader("Búsqueda y Evaluación por Unidad Educativa (AMIE)")

  col1, col2, col3 = st.columns(3)
  col1.metric("Planteles Analizados", len(df_final))
  col2.metric(
      "Tasa Promedio Abandono", f"{df_final['Tasa_Abandono'].mean()*100:.2f}%"
  )
  col3.metric("Niveles de Riesgo", "Bajo (0), Medio (1), Alto (2)")

  st.divider()

  amie_seleccionado = st.selectbox("Seleccione el código AMIE del plantel:", df_final["AMIE"].unique())
  registro = df_final[df_final["AMIE"] == amie_seleccionado].iloc[0]

  riesgo_map = {
      0: ("🟢 BAJO RIESGO DE DESERCIÓN", "st.success"),
      1: ("🟡 RIESGO MEDIO DE DESERCIÓN", "st.warning"),
      2: ("🔴 ALTO RIESGO DE DESERCIÓN", "st.error"),
  }

  etiqueta, tipo_alerta = riesgo_map[registro["NivelRiesgoDesercion"]]

  st.markdown(f"### Nivel de Alerta: **{etiqueta}**")
  if registro["NivelRiesgoDesercion"] == 0:
    st.success(
        f"La institución {amie_seleccionado} mantiene niveles estables de"
        " retención estudiantil."
    )
  elif registro["NivelRiesgoDesercion"] == 1:
    st.warning(
        f"La institución {amie_seleccionado} presenta fluctuaciones moderadas"
        " que requieren monitoreo."
    )
  else:
    st.error(
        f"Alerta preventiva: La institución {amie_seleccionado} requiere"
        " intervención pedagógica inmediata."
    )

  st.dataframe(
      df_final[[
          "AMIE",
          "Total_Estudiantes",
          "Abandono",
          "Tasa_Abandono",
          "NivelRiesgoDesercion",
      ]].head(15),
      use_container_width=True,
  )

with tab2:
  st.subheader(
      "Evaluación Comparativa de Desempeño (MLP Propio vs. 5 Modelos de Línea"
      " Base)"
  )

  y_pred_mlp = mlp.predecir(X_test)
  acc_mlp = accuracy_score(y_test, y_pred_mlp)
  f1_mlp = f1_score(y_test, y_pred_mlp, average="macro", zero_division=0)

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

  tabla_completa = pd.concat([fila_mlp, df_baseline], ignore_index=True)

  st.dataframe(tabla_completa, use_container_width=True)

  st.bar_chart(
      data=tabla_completa,
      x="Modelo",
      y="F1-Score (Macro)",
      color="#1f77b4",
  )

with tab3:
  st.subheader("Análisis de Explicabilidad con Valores SHAP")
  st.write(
      "Visualización del impacto y peso de las características sociodemográficas e"
      " institucionales sobre las predicciones del Perceptrón Multicapa."
  )

  if st.button("Generar Gráfico SHAP"):
    with st.spinner("Calculando valores de Shapley sobre el modelo MLP..."):
      explainer = SHAPExplainer(
          mlp.model.predict, X_train, preprocesador.feature_names
      )
      fig = explainer.generar_grafico_resumen(X_test, n_samples=15)
      st.pyplot(fig)