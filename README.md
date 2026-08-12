# Proyecto Deserción Escolar MINEDUC

Este proyecto implementa una solución de Inteligencia Artificial para predecir la deserción escolar a partir de datos del Ministerio de Educación de Ecuador (MINEDUC).

## Estructura del Proyecto

```
proyecto-desercion-mineduc/
│
├── data/                                 # Datasets del MINEDUC
│   ├── raw/                              # Archivos Excel originales descargados
│   │   ├── 1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx
│   │   └── 2Registro-Administrativo-Historico_2009-2024-Fin.xlsx
│   └── processed/                        # Datos procesados y limpios (generados por ETL)
│       └── dataset_consolidado.csv
│
├── models/                               # Modelos entrenados y serializados
│   ├── mlp_model.h5                      # Modelo propio (Perceptrón Multicapa)
│   ├── baseline_models.pkl               # Los 5 modelos de línea base
│   └── scaler_encoder.pkl                # Objetos StandardScaler y OneHotEncoder
│
├── src/                                  # Código fuente modular en Python
│   ├── __init__.py
│   ├── data_preprocessing.py            # Clase DataPreprocessor (ETL, merge, encodings)
│   ├── mlp_classifier.py                # Clase MLPClassifier (Red neuronal propia en Keras)
│   ├── baseline_evaluator.py            # Clase BaselineEvaluator (5 modelos de librería)
│   └── shap_explainer.py                # Clase SHAPExplainer (Explicabilidad)
│
├── notebooks/                            # Experimentación y Pruebas
│   └── Entrenatorio_y_Evaluacion.ipynb   # Notebook completo para Colab/Jupyter
│
├── app.py                                # Interfaz gráfica interactiva con Streamlit
├── requirements.txt                      # Librerías necesarias para correr el proyecto
└── README.md                             # Instrucciones de ejecución
```

## Instrucciones de Ejecución

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar la aplicación interactiva de Streamlit**:
   ```bash
   streamlit run app.py
   ```

3. **Ejecutar el cuaderno de entrenamiento**:
   Abrir `notebooks/Entrenatorio_y_Evaluacion.ipynb` en Jupyter Notebook, JupyterLab o Google Colab.
