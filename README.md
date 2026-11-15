# Análisis de Datos y Machine Learning de Fuga de Clientes en Telecomunicaciones

Proyecto académico de análisis de datos y Machine Learning aplicado al abandono de clientes (*churn*) en una empresa de telecomunicaciones. El trabajo utiliza el conjunto de datos **Telco Customer Churn** para explorar el comportamiento de los clientes y desarrollar modelos que aborden tres tareas: clasificación del abandono, estimación de la permanencia y segmentación de clientes.

En la parte de clasificación se estudian principalmente los modelos de ensamble **Random Forest** y **XGBoost**. Los árboles de decisión se utilizan como modelos de referencia para comparar su desempeño.

## Estructura del repositorio

```text
.
├── data/                  # Datos del proyecto y documentación asociada
├── docs/                  # Informe técnico y documentación del proyecto
├── modelos/               # Modelos entrenados y archivos relacionados
├── notebooks/             # Notebook principal de análisis y modelado
│   └── analisis_modelado_telco_churn.ipynb
├── resultados/            # Resultados organizados por tipo de análisis
│   ├── clasificacion/
│   ├── regresion/
│   └── segmentacion/
├── src/                   # Código Python reutilizable
│   ├── feature_engineering.py
│   └── visualizaciones.py
├── requirements.txt       # Dependencias de Python
└── README.md
```

## Requisitos de Software

| Función | Tecnologías |
|---|---|
| **Lenguaje Core** | Python 3.12+ |
| **Manipulación y Análisis** | `pandas`, `numpy`, `scipy` |
| **Modelado de ML** | `scikit-learn`, `xgboost`, `kneed` |
| **Visualización** | `matplotlib`, `seaborn` |
| **Entorno de Experimentación** | `JupyterLab`, `ipykernel` |
| **Control de Versiones** | `Git` |

- Bibliotecas de Python y versiones indicadas en `requirements.txt`.

## Instalación

Clona el repositorio:

```bash
git clone https://github.com/david-ramirezl/telco_customer-Modelos-de-Machine-Learning.git
cd telco_customer-Modelos-de-Machine-Learning
```

Crea y activa un entorno virtual. En Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instala las dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Ejecución

Desde la raíz del repositorio y con el entorno virtual activado, inicia JupyterLab:

```bash
jupyter lab
```

Abre el notebook `notebooks/analisis_modelado_telco_churn.ipynb` y ejecuta sus celdas en orden. El notebook contiene el flujo principal de análisis, preparación de datos y modelado.

## Fuente de datos

El proyecto utiliza el conjunto de datos **Telco Customer Churn**. El archivo CSV se obtiene desde el siguiente repositorio de GitHub:

[Telco_Customer_Churn_Dataset.csv](https://raw.githubusercontent.com/david-ramirezl/data/refs/heads/main/Telco_Customer_Churn_Dataset.csv)

El notebook utiliza esta fuente para cargar los datos de entrada.

## Documentación

La carpeta `docs/` contiene el informe técnico y documentos complementarios del proyecto. Las carpetas `data/`, `notebooks/`, `resultados/` y `src/` incluyen sus respectivos archivos de apoyo o documentación cuando corresponde.

## Autores

- Felipe Huincaman
- David Ramírez Luna

Proyecto desarrollado en el contexto de la asignatura **Machine Learning**.
