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
    """Carga y fusiona datasets asegurando identificadores limpios y sufijos claros para prevenir data leakage."""
    df_inicio = pd.read_excel(self.ruta_inicio, nrows=sample_size)
    df_fin = pd.read_excel(self.ruta_fin, nrows=sample_size)

    # Limpieza de nombres de columnas
    df_inicio.columns = df_inicio.columns.str.strip()
    df_fin.columns = df_fin.columns.str.strip()

    # Estandarización de la clave AMIE
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

    # Preservar Año_lectivo para la partición temporal
    on_cols = ["AMIE"]
    if "Año_lectivo" in df_inicio.columns and "Año_lectivo" in df_fin.columns:
      df_inicio["Año_lectivo"] = (
          df_inicio["Año_lectivo"].astype(str).str.strip()
      )
      df_fin["Año_lectivo"] = df_fin["Año_lectivo"].astype(str).str.strip()
      on_cols.append("Año_lectivo")

    # MERGE con sufijos explícitos (_inicio vs _fin)
    df_merged = pd.merge(
        df_inicio, df_fin, on=on_cols, how="inner", suffixes=("_inicio", "_fin")
    )

    if len(df_merged) == 0:
      df_inicio_sub = df_inicio.head(1000)
      df_fin_sub = df_fin[df_fin["AMIE"].isin(df_inicio_sub["AMIE"])]
      df_merged = pd.merge(
          df_inicio_sub,
          df_fin_sub,
          on="AMIE",
          how="inner",
          suffixes=("_inicio", "_fin"),
      )

    return df_merged

  def limpiar_y_calcular_abandono(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la Tasa de Abandono desde las variables de Fin de Año y aísla la variable objetivo."""
    df = df.dropna(how="all", axis=1)

    col_estudiantes_fin = next(
        (
            c
            for c in df.columns
            if ("total_estudiantes" in c.lower() or "estudiantes" in c.lower())
            and ("fin" in c.lower() or c.endswith("_fin"))
        ),
        None,
    )
    if not col_estudiantes_fin:
      col_estudiantes_fin = next(
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

    if not col_estudiantes_fin or not col_abandono:
      raise KeyError(
          "No se encontraron las columnas requeridas para la etiqueta"
          " objetivo."
      )

    df["Total_Estudiantes_Fin"] = pd.to_numeric(
        df[col_estudiantes_fin], errors="coerce"
    )
    df["Abandono"] = pd.to_numeric(df[col_abandono], errors="coerce")

    df = df[
        (df["Total_Estudiantes_Fin"] > 0)
        & (df["Abandono"].notnull())
        & (df["Total_Estudiantes_Fin"].notnull())
    ].copy()

    df["Tasa_Abandono"] = df["Abandono"] / df["Total_Estudiantes_Fin"]
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
        return 0
      elif tasa <= q_high:
        return 1
      else:
        return 2

    df["NivelRiesgoDesercion"] = df["Tasa_Abandono"].apply(asignar_clase)
    return df

  def transformar_caracteristicas(
      self, df: pd.DataFrame, is_training: bool = True
  ):
    """Transforma ÚNICAMENTE variables de INICIO de año lectivo para construir la matriz X y evitar Data Leakage."""
    cols_categoricas = [
        "Sostenimiento",
        "Área",
        "Jornada",
        "Regimen_Escolar",
        "Jurisdiccion",
        "Modalidad",
    ]

    cols_numericas_candidatas = [
        "Total_Docentes",
        "Total_Administrativos",
        "Total_Estudiantes_inicio",
        "Total_Estudiantes",
        "Estudiantes_con_discapacidad",
        "EMestiza",
        "EIndigena",
        "EMontubio",
        "EAfroecuatoriano",
        "EBlanca",
        "EKIchwa",
    ]

    cols_prohibidas_leakage = [
        "Abandono",
        "Promovidos",
        "No_promovidos",
        "Total_Estudiantes_fin",
        "Tasa_Abandono",
        "NivelRiesgoDesercion",
        "promovidos",
        "no_promovidos",
    ]

    cols_cat_existentes = [
        c
        for c in cols_categoricas
        if c in df.columns and c not in cols_prohibidas_leakage
    ]

    cols_num_existentes = []
    for c in cols_numericas_candidatas:
      if c in df.columns and c not in cols_prohibidas_leakage:
        if c == "Total_Estudiantes" and "Total_Estudiantes_inicio" in df.columns:
          continue
        cols_num_existentes.append(c)

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

  def dividir_por_tiempo(self, df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """Aplica la partición cronológica (Time-based Split) para evitar sesgo temporal."""
    if "Año_lectivo" not in df.columns:
      corte = int(len(df) * 0.8)
      return (
          X[:corte],
          X[corte:],
          y[:corte],
          y[corte:],
          "Partición secuencial (80% anterior / 20% reciente)",
      )

    anios_unicos = sorted(df["Año_lectivo"].unique())
    if len(anios_unicos) <= 1:
      corte = int(len(df) * 0.8)
      return (
          X[:corte],
          X[corte:],
          y[:corte],
          y[corte:],
          f"Partición por ordenamiento de lote ({anios_unicos[0]})",
      )

    anio_test = anios_unicos[-1]
    mask_train = df["Año_lectivo"] < anio_test
    mask_test = df["Año_lectivo"] >= anio_test

    if mask_test.sum() < 20 and len(anios_unicos) > 2:
      anio_test = anios_unicos[-2]
      mask_train = df["Año_lectivo"] < anio_test
      mask_test = df["Año_lectivo"] >= anio_test

    X_train, y_train = X[mask_train], y[mask_train]
    X_test, y_test = X[mask_test], y[mask_test]

    info_split = f"Train: < {anio_test} | Test: >= {anio_test}"
    return X_train, X_test, y_train, y_test, info_split