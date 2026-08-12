import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:

  def __init__(self, ruta_inicio: str, ruta_fin: str):
    self.ruta_inicio = ruta_inicio
    self.ruta_fin = ruta_fin
    self.scaler = StandardScaler()
    self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    self.feature_names = []

  def cargar_y_fusionar_datasets(
      self, sample_size: int = 5000
  ) -> pd.DataFrame:
    """Carga los Excel usando un límite de filas (sample_size) para evitar bloqueos por memoria/XML

    y realiza el merge de forma segura.
    """
    print(f"⚡ Leyendo muestra de {sample_size} filas de cada Excel...")

    # Leemos con nrows directamente desde Excel para que sea instantáneo
    df_inicio = pd.read_excel(self.ruta_inicio, nrows=sample_size)
    df_fin = pd.read_excel(self.ruta_fin, nrows=sample_size)

    # Limpiar nombres de columnas
    df_inicio.columns = df_inicio.columns.str.strip()
    df_fin.columns = df_fin.columns.str.strip()

    # Estandarizar nombre de columna clave AMIE
    for col in [
        "Codigo_Institucion",
        "CODIGO_INSTITUCION",
        "Codigo_institucion",
        "amie",
    ]:
      if col in df_inicio.columns:
        df_inicio.rename(columns={col: "AMIE"}, inplace=True)
      if col in df_fin.columns:
        df_fin.rename(columns={col: "AMIE"}, inplace=True)

    df_inicio["AMIE"] = df_inicio["AMIE"].astype(str).str.strip()
    df_fin["AMIE"] = df_fin["AMIE"].astype(str).str.strip()

    # Unir por AMIE (si hay coincidencia entre las muestras)
    df_merged = pd.merge(
        df_inicio, df_fin, on="AMIE", how="inner", suffixes=("_inicio", "_fin")
    )

    # Si por orden de filas no coinciden los AMIE de la muestra, hacemos un outer o tomamos coincidencias
    if len(df_merged) == 0:
      print(
          "⚠️ Advertencia: No coincidieron AMIEs en la muestra inicial. Realizando alineación por intersección..."
      )
      amies_comunes = set(df_inicio["AMIE"]).intersection(set(df_fin["AMIE"]))

      if not amies_comunes:
        # En caso de que los primeros N AMIE no coincidan por diferencia de ordenamiento:
        df_inicio_sub = df_inicio.head(1000)
        df_fin_sub = df_fin[df_fin["AMIE"].isin(df_inicio_sub["AMIE"])]
        df_merged = pd.merge(
            df_inicio_sub,
            df_fin_sub,
            on="AMIE",
            how="inner",
            suffixes=("_inicio", "_fin"),
        )
      else:
        df_inicio_filt = df_inicio[df_inicio["AMIE"].isin(amies_comunes)]
        df_fin_filt = df_fin[df_fin["AMIE"].isin(amies_comunes)]
        df_merged = pd.merge(
            df_inicio_filt,
            df_fin_filt,
            on="AMIE",
            how="inner",
            suffixes=("_inicio", "_fin"),
        )

    return df_merged

  def limpiar_y_calcular_abandono(self, df: pd.DataFrame) -> pd.DataFrame:
    """Limpia nulos, ajusta nombres de columnas y calcula la Tasa de Abandono."""
    df = df.dropna(how="all", axis=1)

    # Detección inteligente de la columna de estudiantes y abandono
    col_estudiantes = next(
        (
            c
            for c in df.columns
            if "total_estudiantes" in c.lower() or "estudiantes" in c.lower()
        ),
        None,
    )
    col_abandono = next(
        (c for c in df.columns if "abandono" in c.lower()), None
    )

    if not col_estudiantes or not col_abandono:
      raise KeyError(
          f"No se encontraron columnas requeridas. Disponibles: {list(df.columns[:10])}"
      )

    df["Total_Estudiantes"] = pd.to_numeric(
        df[col_estudiantes], errors="coerce"
    )
    df["Abandono"] = pd.to_numeric(df[col_abandono], errors="coerce")

    # Filtrar registros válidos
    df = df[
        (df["Total_Estudiantes"] > 0)
        & (df["Abandono"].notnull())
        & (df["Total_Estudiantes"].notnull())
    ].copy()

    # Tasa de Abandono
    df["Tasa_Abandono"] = df["Abandono"] / df["Total_Estudiantes"]
    df["Tasa_Abandono"] = df["Tasa_Abandono"].clip(0.0, 1.0)
    return df

  def discretizar_riesgo(self, df: pd.DataFrame) -> pd.DataFrame:
    """Categoriza la Tasa de Abandono en Bajo (0), Medio (1) y Alto (2)."""
    quantiles = df["Tasa_Abandono"].quantile([0.33, 0.66]).values
    q_low, q_high = quantiles[0], quantiles[1]

    if q_low == q_high:
      q_low, q_high = 0.02, 0.08

    def asignar_clase(tasa):
      if tasa <= q_low:
        return 0  # Bajo
      elif tasa <= q_high:
        return 1  # Medio
      else:
        return 2  # Alto

    df["NivelRiesgoDesercion"] = df["Tasa_Abandono"].apply(asignar_clase)
    return df

  def transformar_caracteristicas(
      self, df: pd.DataFrame, is_training: bool = True
  ):
    """Aplica OneHotEncoding a variables categóricas y StandardScaler a numéricas."""
    cols_categoricas = [
        "Sostenimiento",
        "Área",
        "Jornada",
        "Regimen_Escolar",
        "Jurisdiccion",
        "Modalidad",
    ]
    cols_numericas = [
        "Total_Docentes",
        "Total_Administrativos",
        "Total_Estudiantes",
        "Estudiantes_con_discapacidad",
        "EMestiza",
        "EIndigena",
        "EMontubio",
        "EAfroecuatoriano",
        "EBlanca",
        "EKIchwa",
    ]

    cols_cat_existentes = [c for c in cols_categoricas if c in df.columns]
    cols_num_existentes = [c for c in cols_numericas if c in df.columns]

    df[cols_num_existentes] = df[cols_num_existentes].fillna(0)
    df[cols_cat_existentes] = df[cols_cat_existentes].fillna("Desconocido")

    if is_training:
      cat_encoded = self.encoder.fit_transform(df[cols_cat_existentes])
      num_scaled = self.scaler.fit_transform(df[cols_num_existentes])
    else:
      cat_encoded = self.encoder.transform(df[cols_cat_existentes])
      num_scaled = self.scaler.transform(df[cols_num_existentes])

    X_processed = np.hstack([num_scaled, cat_encoded])
    y = df["NivelRiesgoDesercion"].values

    encoded_cat_names = list(
        self.encoder.get_feature_names_out(cols_cat_existentes)
    )
    self.feature_names = cols_num_existentes + encoded_cat_names

    return X_processed, y