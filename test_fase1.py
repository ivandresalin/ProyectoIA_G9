import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import DataPreprocessor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("🚀 Iniciando prueba de la Fase 1...")

ruta_inicio = (
    "data/raw/1Registro-Administrativo-Historico_2009-202X-Inicio.xlsx"
)
ruta_fin = "data/raw/2Registro-Administrativo-Historico_2009-2024-Fin.xlsx"

try:
  preprocesador = DataPreprocessor(ruta_inicio, ruta_fin)

  # Pedimos muestra de 3000 filas (se procesa en 2 segundos)
  df_raw = preprocesador.cargar_y_fusionar_datasets(sample_size=3000)
  print(f"   └─ Registros fusionados obtenidos: {len(df_raw)}")

  print("🧹 Limpiando datos y calculando Tasa de Abandono...")
  df_clean = preprocesador.limpiar_y_calcular_abandono(df_raw)

  print("🏷️ Discretizando niveles de riesgo...")
  df_final = preprocesador.discretizar_riesgo(df_clean)

  print("⚡ Transformando características (Scaling y Encoding)...")
  X, y = preprocesador.transformar_caracteristicas(df_final, is_training=True)

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42, stratify=y
  )

  print("\n" + "=" * 50)
  print("✅ ¡FASE 1 COMPLETADA CON ÉXITO!")
  print("=" * 50)
  print(f"📐 Forma de X_train (filas, columnas de entrada): {X_train.shape}")
  print(f"📐 Forma de X_test:  {X_test.shape}")
  print("\n📊 Distribución de clases (0: Bajo, 1: Medio, 2: Alto):")
  print(pd.Series(y).value_counts().sort_index())

except Exception as e:
  print(f"\n❌ Se produjo un error: {e}")