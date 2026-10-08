import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

"""Funciones para crear visualizaciones del análisis exploratorio y evaluación.

Las funciones de este módulo reciben un DataFrame con la estructura del dataset
de clientes y generan gráficos relacionados con la variable objetivo "Churn".
Las figuras se devuelven para poder mostrarlas desde el notebook, permitiendo 
su guardado manual posterior si se desea.
"""


def graficar_distribucion_churn(df):
    """Crea un gráfico de barras con la distribución de clientes según Churn.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene la columna "Churn" con los valores "Yes" y
        "No".

    Returns
    -------
    tuple
        La figura de Matplotlib y el eje que contiene el gráfico.
    """
    datos = df.copy()
    datos["Churn"] = datos["Churn"].map({"Yes": "Sí", "No": "No"})

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(
        data=datos,
        x="Churn",
        hue="Churn",
        order=["No", "Sí"],
        palette=["#4C72B0", "#C44E52"],
        width=0.65,
        legend=False,
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)

    ax.set_title("Distribución de la variable objetivo", fontsize=16, fontweight="bold")
    ax.set_xlabel("Abandono del servicio")
    ax.set_ylabel("Cantidad de clientes")
    ax.set_ylim(0, datos["Churn"].value_counts().max() * 1.1)

    return fig, ax


def graficar_curvas_densidad_churn(df):
    """Crea curvas KDE para variables numéricas según Churn.

    La figura contiene las distribuciones de ``tenure``, ``MonthlyCharges`` y
    ``TotalCharges`` organizadas en una cuadrícula de dos columnas. Las curvas
    se diferencian según si el cliente abandonó o no el servicio.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables numéricas y la columna ``Churn``.

    Returns
    -------
    tuple
        La figura de Matplotlib (cuadrícula) y el arreglo de ejes utilizados.
    """
    datos = df.copy()
    datos["Churn"] = datos["Churn"].map({"Yes": "Sí", "No": "No"})

    variables = {
        "tenure": ("Antiguedad del cliente", "Antiguedad (meses)"),
        "MonthlyCharges": ("Cargos mensuales", "Cargos mensuales ($)"),
        "TotalCharges": ("Cargos totales", "Cargos totales ($)")
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Distribución de variables numéricas según abandono",
        fontsize=16,
        fontweight="bold"
    )
    axes = axes.ravel()

    for indice, (variable, (titulo, etiqueta_x)) in enumerate(variables.items()):
        sns.kdeplot(
            data=datos,
            x=variable,
            hue="Churn",
            fill=True,
            palette=["#4C72B0", "#C44E52"],
            common_norm=False,
            multiple="layer",
            alpha=0.45,
            ax=axes[indice]
        )
        axes[indice].set_title(titulo)
        axes[indice].set_xlabel(etiqueta_x)
        axes[indice].set_ylabel("Densidad")

    axes[3].set_visible(False)
    fig.subplots_adjust(wspace=0.3, hspace=0.35, top=0.88)

    return fig, axes


GRUPOS_CATEGORICOS = {
    "Información del cliente": ["gender", "SeniorCitizen", "Partner", "Dependents"],
    "Servicios contratados": ["PhoneService", "MultipleLines", "InternetService", "StreamingTV", "StreamingMovies"],
    "Seguridad y soporte": ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"],
    "Información de pago": ["Contract", "PaperlessBilling", "PaymentMethod"]
}

NOMBRES_VARIABLES = {
    "gender": "Género",
    "SeniorCitizen": "Adulto mayor",
    "Partner": "Pareja",
    "Dependents": "Personas dependientes",
    "PhoneService": "Servicio telefónico",
    "MultipleLines": "Líneas múltiples",
    "InternetService": "Servicio de internet",
    "StreamingTV": "Televisión por streaming",
    "StreamingMovies": "Películas por streaming",
    "OnlineSecurity": "Seguridad en línea",
    "OnlineBackup": "Respaldo en línea",
    "DeviceProtection": "Protección del dispositivo",
    "TechSupport": "Soporte técnico",
    "Contract": "Tipo de contrato",
    "PaperlessBilling": "Facturación electrónica",
    "PaymentMethod": "Método de pago"
}

TRADUCCIONES_CATEGORIAS = {
    "Female": "Mujer",
    "Male": "Hombre",
    "Yes": "Sí",
    "No": "No",
    "No phone service": "Sin servicio telefónico",
    "Fiber optic": "Fibra óptica",
    "No internet service": "Sin servicio de internet",
    "Month-to-month": "Mes a mes",
    "One year": "Un año",
    "Two year": "Dos años",
    "Electronic check": "Cheque electrónico",
    "Mailed check": "Cheque enviado por correo",
    "Bank transfer (automatic)": "Transferencia bancaria (automática)",
    "Credit card (automatic)": "Tarjeta de crédito (automática)"
}


def _crear_countplot(datos, variable, ax):
    """Dibuja un countplot categórico con formato y etiquetas consistentes.

    Parameters
    ----------
    datos : pandas.DataFrame
        DataFrame preparado con las etiquetas de ``Churn`` traducidas.
    variable : str
        Nombre de la variable categórica que se representará en el eje X.
    ax : matplotlib.axes.Axes
        Eje donde se dibujará el gráfico.
    """
    sns.countplot(
        data=datos,
        x=variable,
        hue="Churn",
        hue_order=["No", "Sí"],
        palette=["#4C72B0", "#C44E52"],
        width=0.65,
        ax=ax
    )

    etiquetas = [
        TRADUCCIONES_CATEGORIAS.get(etiqueta.get_text(), etiqueta.get_text())
        for etiqueta in ax.get_xticklabels()
    ]
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=15 if variable == "PaymentMethod" else 0)
    ax.set_title(NOMBRES_VARIABLES[variable])
    ax.set_xlabel("")
    ax.set_ylabel("Cantidad de clientes")
    ax.margins(y=0.12)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)


def graficar_grupo_categorico_churn(df, nombre_grupo):
    """Genera visualizaciones categóricas para un grupo específico según el abandono.

    Crea una sábana con dos gráficos por fila para las variables del grupo solicitado.
    Los títulos y categorías se traducen al español, se muestran los conteos sobre 
    las barras y solo el gráfico de ``PaymentMethod`` se presenta sin leyenda.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables categóricas y la columna ``Churn``.
    nombre_grupo : str
        Nombre del grupo específico a graficar (debe coincidir con una llave de GRUPOS_CATEGORICOS).

    Returns
    -------
    matplotlib.figure.Figure
        La figura que agrupa los gráficos generados para el grupo indicado.
    """
    if nombre_grupo not in GRUPOS_CATEGORICOS:
        raise ValueError(f"El grupo '{nombre_grupo}' no es válido. Opciones: {list(GRUPOS_CATEGORICOS.keys())}")

    variables = GRUPOS_CATEGORICOS[nombre_grupo]
    
    datos = df.copy()
    datos["Churn"] = datos["Churn"].map({"Yes": "Sí", "No": "No"})

    cantidad_filas = (len(variables) + 1) // 2
    figura_grupo, ejes = plt.subplots(
        cantidad_filas,
        2,
        figsize=(15, 5.2 * cantidad_filas)
    )
    ejes = ejes.reshape(-1)
    
    figura_grupo.suptitle(
        f"Distribución de {nombre_grupo} según abandono",
        fontsize=16,
        fontweight="bold"
    )

    for indice, variable in enumerate(variables):
        _crear_countplot(datos, variable, ejes[indice])
        
        # Ocultar la leyenda únicamente en PaymentMethod
        if variable == "PaymentMethod":
            ejes[indice].get_legend().remove()
        else:
            ejes[indice].legend(title="Abandono")

    for eje in ejes[len(variables):]:
        eje.set_visible(False)

    figura_grupo.subplots_adjust(
        wspace=0.3,
        hspace=0.4,
        top=0.88,
        bottom=0.1
    )
    
    return figura_grupo


def plot_matriz_confusion(y_true, y_pred, nombre_modelo="", clases=['No Abandona', 'Sí Abandona'], cmap='Blues'):
    """
    Genera y grafica una matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True, 
                xticklabels=clases, yticklabels=clases, annot_kws={"size": 14})
    
    titulo = f'Matriz de Confusión - {nombre_modelo}' if nombre_modelo else 'Matriz de Confusión'
    plt.title(titulo, fontsize=15, pad=15, fontweight='bold')
    
    plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_curva_roc(y_true, y_proba, nombre_modelo=""):
    """
    Grafica la curva ROC y calcula el AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#ff7f0e', lw=2.5, label=f'Curva ROC (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Modelo Aleatorio (0.5)')
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    
    # Adaptación dinámica del título
    titulo = f'Curva ROC - {nombre_modelo}' if nombre_modelo else 'Curva ROC'
    plt.title(titulo, fontsize=15, pad=15, fontweight='bold')
    
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()