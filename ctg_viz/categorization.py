"""
Módulo de categorización de variables y análisis de completitud.

Incluye:
- Revisión de completitud y nulos
- Estadísticos descriptivos por columna
- Clasificación automática de variables en continuas, discretas y categóricas
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List


def check_data_completeness_davidmaganacelis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen de la completitud y estadísticas de cada columna.

    Criterios:
        - n_nulls: Conteo de valores nulos.
        - pct_completeness: Porcentaje de datos no nulos.
        - dtype: Tipo de dato original.
        - mean, std, min, 25%, 50%, 75%, max: Estadísticos descrip. numéricos.

    Args:
        df (pd.DataFrame): DataFrame a analizar.

    Returns:
        pd.DataFrame: Resumen completo de cada columna.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[
                "n_nulls",
                "pct_completeness",
                "dtype",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        )

    resumen = pd.DataFrame({
        "n_nulls": df.isna().sum(),
        "pct_completeness": (1 - df.isna().sum() / len(df)) * 100,
        "dtype": df.dtypes.astype(str)
    })

    # Obtener estadísticas sólo para columnas numéricas
    stats = df.describe().T.reindex(resumen.index)

    # Combinar
    resumen = resumen.join(stats, how="left")
    resumen["pct_completeness"] = resumen["pct_completeness"].round(2)

    return resumen


def clasificar_columnas(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Clasifica automáticamente columnas en continuas, discretas y categóricas.

    Criterios:
        - Continuas: Numéricas con más de 10 valores únicos.
        - Discretas: Numéricas con 10 o menos valores únicos.
        - Categóricas: Cualquier columna no numérica.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        dict: Diccionario con listas de columnas por categoría.
    """

    col_numericas = df.select_dtypes(include="number").columns.tolist()
    col_categoricas = df.select_dtypes(exclude="number").columns.tolist()

    continuas = []
    discretas = []

    for col in col_numericas:
        if df[col].nunique() > 10:
            continuas.append(col)
        else:
            discretas.append(col)

    return {
        "continuas": continuas,
        "discretas": discretas,
        "categoricas": col_categoricas
    }


def categorization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un reporte unificado con la clasificación y completitud por columna.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: Resumen profesional para reporte.
    """
    resumen = check_data_completeness_davidmaganacelis(df)
    clasificacion = clasificar_columnas(df)

    resumen["clasificacion"] = "categórica"  # Default

    for col in clasificacion["continuas"]:
        resumen.loc[col, "clasificacion"] = "continua"

    for col in clasificacion["discretas"]:
        resumen.loc[col, "clasificacion"] = "discreta"

    return resumen
