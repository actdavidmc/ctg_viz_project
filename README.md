# ctg_viz_project

Librería personalizada para análisis exploratorio y visualización del dataset CTG (Cardiotocography).
Incluye módulos para preprocesamiento, categorización y creación de gráficos, así como un conjunto de pruebas automáticas con pytest.

---

## Estructura del proyecto

ctg_viz_project/
│
├── ctg_viz/
│   ├── preprocessing.py
│   ├── categorization.py
│   ├── plots/
│   │   ├── histograms.py
│   │   ├── barplots.py
│   │   └── ...
│   └── __init__.py
│
├── data/
│   └── CTG.csv
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_categorization.py
│   ├── test_plots.py
│   └── conftest.py
│
├── requirements.txt
└── README.md

---

## Descripción de los módulos

### 1. preprocessing.py

Implementa el pipeline completo de preprocesamiento:

- Carga de archivos CSV o Excel.
- Estandarización de nombres de columnas.
- Conversión de tipos numéricos y fechas.
- Eliminación de columnas con alto porcentaje de valores nulos.
- Imputación de valores faltantes (mean, median o KNN).
- Detección y tratamiento de outliers (IQR o Z-score).
- Winsorización o eliminación de filas con outliers.

Función principal:

from ctg_viz.preprocessing import preprocessing
df = preprocessing("data/CTG.csv")

---

### 2. categorization.py

Incluye funciones para:

- Medir completitud de datos.
- Clasificar columnas en continuas, discretas o categóricas.
- Generar un resumen global del dataset.

Ejemplo:

from ctg_viz.categorization import check_data_completeness, clasificacion_variables

---

### 3. plots/

Incluye funciones de visualización:

- Histogramas con o sin KDE.
- Facetado por variable categórica.
- Gráficas de barras para variables categóricas.

Ejemplo:

from ctg_viz.plots.histograms import plot_histogram
plot_histogram(df, columnas=["LB", "FM"])

---

## Instalación y entorno

1. Crear y activar entorno virtual:

python3 -m venv .venv
source .venv/bin/activate

2. Instalar dependencias:

pip install -r requirements.txt

---

## Ejecución de tests

Todos los tests:

pytest -vv

Por archivo:

pytest tests/test_preprocessing.py
pytest tests/test_categorization.py
pytest tests/test_plots.py

Test específico:

pytest tests/test_preprocessing.py::test_convertir_tipos

---

## Uso del proyecto

1. Preprocesamiento:

from ctg_viz.preprocessing import preprocessing
df = preprocessing("data/CTG.csv", metodo_imputacion="mean")

2. Categorización:

from ctg_viz.categorization import resumen_categorizacion
summary = resumen_categorizacion(df)
print(summary)

3. Visualizaciones:

from ctg_viz.plots.histograms import plot_histogram
plot_histogram(df, columnas=["LB"], kde=True)

---

## Objetivo del proyecto

Este proyecto forma parte de una práctica de análisis exploratorio de datos en el contexto de un diplomado en ciencia de datos.
Demuestra:

- Buenas prácticas de ingeniería de datos.
- Modularidad y reutilización del código.
- Correcta implementación de pruebas unitarias.
- Presentación profesional de análisis y visualizaciones.

---

## Autor

David Magaña Celis
Diplomado en Ciencia de Datos
