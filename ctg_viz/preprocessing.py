"""
Módulo creado para el preprocesamiento de DataFrames.

Incluye:
- Carga de archivos
- Estandarización de columnas
- Conversión de tipos
- Eliminación de columnas con alto porcentaje de nulos
- Imputación de valores faltantes
- Detección y tratamiento de valores atípicos
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import zscore


def cargar_archivo(path: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV o Excel.

    Args:
        path (str): Ruta al archivo.

    Returns:
        pd.DataFrame: Datos cargados.

    Raises:
        ValueError: Si el archivo no puede leerse.
    """
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path)
        else:
            return pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        raise ValueError(f"No se pudo cargar el archivo: {e}")


def estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Estandariza nombres de columnas a un formato limpio.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame con nombres estandarizados.
    """
    df_est = df.copy()
    df_est.columns = (
        df_est.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("-", "_")
            .str.replace(".", "_")
    )
    return df_est


def convertir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los tipos de dato de manera automática.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame con tipos convertidos.
    """
    df_conv = df.copy()
    return df_conv.convert_dtypes()


def eliminar_col_nulos(df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """Elimina columnas donde el porcentaje de nulos supera un umbral.

    Args:
        df (pd.DataFrame): DataFrame original.
        threshold (float): Proporción máxima permitida de nulos.

    Returns:
        pd.DataFrame: DataFrame reducido.
    """
    df_temp = df.copy()
    pnull = df_temp.isna().mean()
    col_drop = pnull[pnull > threshold].index.tolist()
    return df_temp.drop(columns=col_drop)


def imputar_valores(df: pd.DataFrame, metodo: str = "mean", knn_vecinos: int = 5) -> pd.DataFrame:
    """Imputa valores faltantes usando diferentes métodos.

    Args:
        df (pd.DataFrame): DataFrame original.
        metodo (str): mean | median | knn.
        knn_vecinos (int): Vecinos a considerar para KNN.

    Returns:
        pd.DataFrame: DataFrame imputado.
    """
    df_imp = df.copy()
    col_num = df_imp.select_dtypes(include="number").columns
    col_cat = df_imp.select_dtypes(exclude="number").columns

    if metodo in ["mean", "median"]:
        for col in col_num:
            imp = SimpleImputer(strategy=metodo)
            df_imp[[col]] = imp.fit_transform(df_imp[[col]])

    elif metodo == "knn":
        imp = KNNImputer(n_neighbors=knn_vecinos)
        df_imp[col_num] = imp.fit_transform(df_imp[col_num])

    else:
        raise ValueError("Método no válido: mean, median, knn")

    # Categóricas → moda
    if len(col_cat) > 0:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df_imp[col_cat] = imp_cat.fit_transform(df_imp[col_cat])

    return df_imp


def outliers_iqr(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Identifica outliers usando el rango intercuartílico (IQR)."""
    df_temp = df.copy()
    col_num = df_temp.select_dtypes(include="number").columns
    resultados = {}

    for col in col_num:
        q1 = df_temp[col].quantile(0.25)
        q3 = df_temp[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr

        resultados[col] = (df_temp[col] < low) | (df_temp[col] > high)

    return resultados


def outliers_z(df: pd.DataFrame, umbral: float = 3) -> Dict[str, pd.Series]:
    """Identifica outliers usando Z-score."""
    df_temp = df.copy()
    col_num = df_temp.select_dtypes(include="number").columns
    resultados = {}

    for col in col_num:
        z_val = zscore(df_temp[col], nan_policy="omit")
        resultados[col] = np.abs(z_val) > umbral

    return resultados


def tratar_outliers(df: pd.DataFrame, metodo: str = "iqr", accion: str = "winsorize") -> pd.DataFrame:
    """Trata valores atípicos usando IQR o Z-score."""
    df_out = df.copy()

    if metodo == "iqr":
        mascaras = outliers_iqr(df_out)
    elif metodo == "zscore":
        mascaras = outliers_z(df_out)
    else:
        raise ValueError("Método inválido: usa 'iqr' o 'zscore'.")

    if accion == "drop":
        filas = pd.DataFrame(mascaras).any(axis=1)
        return df_out[~filas].reset_index(drop=True)

    if accion == "winsorize":
        col_num = df_out.select_dtypes(include="number").columns

        for col in col_num:
            if metodo == "iqr":
                q1 = df_out[col].quantile(0.25)
                q3 = df_out[col].quantile(0.75)
                iqr = q3 - q1
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
            else:  
                media = df_out[col].mean()
                std = df_out[col].std()
                low = media - 3 * std
                high = media + 3 * std

            df_out[col] = np.clip(df_out[col], low, high)

        return df_out

    raise ValueError("Acción inválida: usa 'winsorize' o 'drop'.")


def preprocessing(
    path: str,
    threshold_nulos: float = 0.2,
    metodo_imputacion: str = "mean",
    metodo_outliers: str = "iqr",
    accion_outliers: str = "drop"
) -> pd.DataFrame:
    """Pipeline de preprocesamiento completo."""
    
    df = cargar_archivo(path)
    df = estandarizar_columnas(df)
    df = convertir_tipos(df)
    df = eliminar_col_nulos(df, threshold_nulos)
    df = imputar_valores(df, metodo=metodo_imputacion)
    df = tratar_outliers(df, metodo=metodo_outliers, accion=accion_outliers)

    return df