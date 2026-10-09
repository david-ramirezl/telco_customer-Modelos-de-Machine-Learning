# Informe Técnico: Implementación de Modelos de Machine Learning en Telecomunicaciones

    - Equipo de Trabajo: Felipe Huincaman, David  Ramírez Luna y Octavio Chávez
    - Asignatura: Machine Learning
    - Docente: Profesora Jazna Meza Hidalgo
    - Fecha de elaboración:

# 1. Problema de negocio
Para las empresas de telecomunicaciones, la retención de clientes es un desafío empresarial crítico. En un mercado donde los usuarios dependen de la conectividad diaria y pueden migrar fácilmente hacia la competencia ante la más mínima interrupción, mala experiencia o una oferta más atractiva, la tasa de abandono (churn) representa uno de los mayores riesgos financieros del sector. Considerando que adquirir un nuevo usuario cuesta hasta cinco veces más que retener a uno actual, no controlar el abandono se traduce en una pérdida directa de ingresos que amenaza la rentabilidad de la compañía.

Para mitigar este impacto, adoptar una gestión preventiva e integral del ciclo de vida del cliente. El principal desafío analítico para el negocio requiere responder a tres preguntas estratégicas fundamentales: ¿Cómo se agrupan nuestros clientes según sus hábitos de consumo y facturación?, ¿Cuáles de ellos están en riesgo inminente de cancelar su servicio?, y ¿Cuánto tiempo de permanencia esperado le queda a cada usuario en la compañía?

Al comprender la diversidad de perfiles mediante la segmentación de la cartera, anticipar de forma precisa las bajas y proyectar el tiempo de vida del cliente, la empresa de telecomunicaciones puede optimizar sus presupuestos de retención. Esto permite a los equipos comerciales priorizar esfuerzos en los usuarios más rentables, diseñar campañas de fidelización personalizadas a las necesidades de cada nicho y mejorar proactivamente la experiencia del servicio, convirtiendo la retención en un pilar estratégico de crecimiento sostenible.

# 2. Objetivos del proyecto
### Objetivo general del proyecto:
Diseñar una estrategia integral basada en datos para mitigar el impacto financiero del abandono de clientes en la empresa de telecomunicaciones, sustentada en la segmentación de los usuarios según sus patrones de comportamiento y consumo, la detección anticipada de futuras cancelaciones y la estimación del tiempo de permanencia esperado para cada cliente, con el fin de diseñar campañas de fidelización personalizadas y optimizar la inversión del equipo de retención.

### Objetivos específicos de etapa de Implementación de Modelos de Machine Learning:
1. Comprensión de la base de clientes: Analizar el comportamiento histórico y las características de los usuarios para identificar los factores clave y los motivos que impulsan las cancelaciones de servicio, asegurando información confiable para la toma de decisiones.
2. Segmentación del mercado: Descubrir y definir perfiles de clientes con comportamientos de consumo y facturación similares, permitiendo a los equipos de marketing diseñar campañas de fidelización específicas y adaptadas a las necesidades de cada arquetipo.
3. Detección de fugas: Identificar de forma anticipada a los clientes que presentan una alta probabilidad de abandonar la compañía, proporcionando al negocio alertas tempranas para intervenir de manera proactiva antes de que se concrete la baja.
4. Proyección de valor y permanencia: Estimar el tiempo de permanencia esperado de los usuarios activos, con el fin de priorizar los presupuestos de retención hacia aquellos clientes que representan una mayor rentabilidad a largo plazo para la empresa.
5. Validación de confiabilidad de la solución: Asegurar la precisión y fiabilidad de las herramientas predictivas desarrolladas, garantizando que las estrategias comerciales y las inversiones en retención se basen en proyecciones certeras que minimicen los costos por falsas alarmas o fugas no detectadas.
6. Garantía de responsabilidad ética: Evaluar el impacto comercial y social de las estrategias propuestas, garantizando el respeto a la privacidad de los clientes, la transparencia en el uso de su información y la ausencia de tratos discriminatorios o sesgados en las campañas de fidelización.

# 3. Definición de KPIs que resolverán el problema de negocio
Para asegurar que los modelos de Machine Learning desarrollados generen un impacto cuantificable en la empresa de telecomunicaciones, el desempeño del proyecto se evaluará mediante los siguientes Indicadores Clave de Desempeño:

### Tasa de Retención Predictiva
Mide el porcentaje de clientes identificados correctamente con alto riesgo de abandono que logran ser retenidos tras la intervención proactiva de la empresa.

- Se obtiene dividiendo la cantidad de clientes que aceptaron la oferta de retención entre el total de usuarios que el modelo predictivo clasificó previamente como propensos a abandonar, expresando el resultado como un porcentaje.
- Valida la efectividad directa del modelo de clasificación, demostrando si la detección anticipada se traduce en una reducción real de la pérdida de clientes.

### Proyección de Ingresos Recuperados
Cuantifica el impacto financiero de retener a un usuario en riesgo, estimando el dinero que ingresará a la compañía gracias a su permanencia.

- Se determina multiplicando la cuota mensual del cliente por la cantidad de meses que el modelo de regresión estima que permanecerá en la compañía (variable Tenure). Luego, se suman estos valores individuales para obtener el total de ingresos asegurados por todos los clientes retenidos.
- Conecta el modelo de regresión con los objetivos financieros, permitiendo a la gerencia priorizar los esfuerzos comerciales hacia los clientes que representan una mayor rentabilidad proyectada a largo plazo.

### Retorno de Inversión (ROI) de la Retención
Evalúa la rentabilidad de las campañas de fidelización al dirigir las ofertas exclusivamente a los clientes que el modelo identifica con riesgo real, comparando lo gastado en descuentos versus lo recuperado.

- Se calcula restando los costos totales de la campaña de retención a los ingresos futuros recuperados. Ese beneficio neto se divide por el costo inicial de la campaña para revelar el porcentaje de ganancia sobre la inversión.
- Demuestra la eficiencia del algoritmo al minimizar los falsos positivos. Al evitar dar descuentos a usuarios que no tenían intención de cancelar el servicio, se optimiza el presupuesto de marketing.

### Índice de Fuga por Clúster
Calcula la proporción de cancelaciones que ocurren dentro de cada segmento específico de clientes, agrupados previamente por sus similitudes de consumo.

- Se obtiene dividiendo la cantidad de clientes que efectivamente abandonaron el servicio dentro de un segmento determinado (Clúster) entre el número total de usuarios que conforman ese mismo grupo, para identificar qué porcentaje de ese nicho específico representa una fuga.
- Valida el modelo de clustering, entregando insights estratégicos para identificar si el problema del abandono afecta a la base completa de manera uniforme o si es una falla estructural de un perfil específico de usuarios.

# 4. Metodología utilizada (CRISP-DM)
Para estructurar y ejecutar este proyecto se adoptó la metodología CRISP-DM, garantizando un desarrollo coherente y alineado con los objetivos de negocio.

1. Comprensión del Negocio: Se diagnosticó el impacto financiero del abandono de clientes en las telecomunicaciones. Para abordarlo, se definieron objetivos analíticos de clasificación, regresión y segmentación, cuyo éxito se medirá mediante KPIs estratégicos de retención y rentabilidad.

2. Comprensión de los Datos: Se importó un conjunto de 7043 registros y 21 características. La exploración inicial reveló un desbalance en la variable objetivo, con una tasa de abandono del 26,5%. Además, se evidenció que los clientes que desertan tienen una permanencia media de solo 17,98 meses frente a los 37,57 meses de los activos, y asumen cargos mensuales más elevados.   

3. Preparación de los Datos: La limpieza y transformación en Python incluyó:

    - Anonimización: Se aplicó un cifrado irreversible SHA-256 a las identificaciones para proteger la privacidad de los usuarios de forma normativa.   
    - Tratamiento de Nulos: Se imputaron con cero los 11 valores nulos encontrados en los cargos totales, ya que correspondían estrictamente a clientes nuevos sin antigüedad.   
    - Auditoría de Calidad: Se verificó la ausencia total de registros duplicados y de valores atípicos en las variables numéricas mediante el método de rango intercuartil.   

4. Modelado: Con la base procesada mediante tuberías de Scikit-Learn, el desarrollo se divide en tres enfoques:

    - Clasificación: Entrenamiento de algoritmos supervisados para predecir la probabilidad de fuga.
    - Regresión: Modelos supervisados para proyectar el ciclo de vida y el tiempo de permanencia del usuario.
    - Segmentación: Técnicas no supervisadas para descubrir grupos poblacionales y detectar anomalías en la base de clientes.

5. Evaluación: El desempeño de los modelos se medirá inicialmente con métricas matemáticas de exactitud y exhaustividad. Luego, este rendimiento se traducirá a los KPIs de negocio para evaluar su eficiencia en el contexto comercial real.   

6. Despliegue: La solución se consolida en una arquitectura reproducible con subdirectorios para datos, cuadernos de experimentación y código fuente. El entregable final consta de los cuadernos de Jupyter debidamente documentados y este informe técnico en formato Markdown.   

# 5. Descripción General y Calidad del Conjunto de Datos
El proyecto utiliza el conjunto de datos "Telco Customer Churn", que registra el historial y la retención de clientes de una empresa de telecomunicaciones. El archivo contiene 7.043 registros (un cliente por fila) y 21 características.

La información se agrupa en cuatro dimensiones principales:   

- Variable de abandono: Indicador que señala si el cliente se dio de baja o canceló su servicio durante el último mes, representado en la columna objetivo Churn.
- Servicios suscritos: Detalle de de productos contratados por cada cliente, lo que incluye servicio telefónico, múltiples líneas, tipo de internet, seguridad en línea, respaldo en la nube, protección de dispositivos, soporte técnico y transmisión (streaming) de TV y películas.
- Información de la cuenta: Atributos relacionados con el estado comercial e historial del cliente, tales como la cantidad de meses que llevan en la compañía (tenure), el tipo de contrato, el método de pago, la adopción de facturación electrónica, los cargos mensuales y los cargos totales acumulados.
- Información demográfica: Características del perfil del cliente, abarcando su género, si es considerado adulto mayor (SeniorCitizen), y su situación familiar respecto a si conviven con pareja o tienen dependientes.

# 6. Preparación y análisis exploratorio de los datos (EDA)
Como paso previo al modelamiento, se auditó y exploró la base de datos para garantizar su calidad y comprender el comportamiento comercial de los clientes.

## 6.1 Auditoría y Calidad del Conjunto de Datos
La revisión técnica de los 7043 registros confirmó una alta integridad estructural:
- Anonimización: Se aplicó la función criptográfica SHA-256 a la identificación del cliente para proteger la privacidad de los usuarios desde el inicio.
- Completitud e Imputación: La calidad es excelente. Solo se encontraron 11 valores nulos en los cargos totales. Al tratarse de clientes nuevos con cero meses de antigüedad, la decisión técnica fue imputarlos directamente con el valor cero.
- Valores Atípicos: La evaluación mediante el método de Rango Intercuartil descartó la presencia de valores atípicos en la antigüedad y en los cargos mensuales y totales.
- Duplicidad: No se detectaron registros duplicados.   

## 6.2 Análisis Descriptivo y Comportamiento del Cliente
La exploración de los datos permitió identificar patrones claros en el perfil, el consumo y las tendencias de cancelación que servirán como base para los futuros modelos.

#### Perfil Demográfico y Distribución de Servicios:
La base está equilibrada por género, con 3555 hombres y 3488 mujeres, y orientada a un público joven, ya que el 83,8% no pertenece al segmento de adultos mayores. El teléfono es el servicio básico indispensable, adoptado por el 90,3% de los usuarios. Comercialmente, el 55% de la cartera prefiere la flexibilidad de los contratos mensuales.

#### Ciclo de Vida y Costos:
- La antigüedad global promedia los 32,37 meses. Los clientes activos alcanzan los 37,57 meses, mientras que las deserciones ocurren tempranamente a los 17,98 meses en promedio.    
- Quienes cancelan asumen tarifas mensuales más altas de $74,44 frente a los $61,27 de los activos. A largo plazo, los usuarios retenidos acumulan un gasto muy superior de $2549 frente a $1531.

#### Factores de Fricción y Retención:
A nivel global, el 26,5% de los usuarios decide cancelar el servicio. El análisis detectó las siguientes correlaciones clave para este abandono:   
- Contratos Cortos: El formato mensual presenta un 42,7% de abandono. En contraste, los compromisos a uno o dos años casi no registran salidas.
- Fricción en Pagos: La facturación electrónica duplica las cancelaciones frente a la tradicional, alcanzando un 33,5%. Asimismo, el pago por cheque electrónico es el riesgo principal con un 45,2% de abandonos.
- Complementos: Añadir seguridad en línea o soporte técnico reduce las cancelaciones del 42% a un 15%. De igual forma, la protección de equipos las baja del 40% al 22%.

# 7. Implementación de modelos de aprendizaje supervisado
## 7.1 Clasificación
### 7.1.1 Preparación para el modelado
#### Criterios de selección de características
La selección de las variables predictoras se basó en una validación matemática y otra visual. Para las variables categóricas, usamos la métrica V de Cramér, considerando que un valor sobre 0.3 es un buen indicador de que la característica realmente afecta la decisión de abandono del cliente. Estos niveles de asociación estadística se detallan en el siguiente gráfico:

![Matriz de Correlación](./resultados/clasificacion/plots/cramers_churn_categoricas.png)

Por el lado de las variables numéricas, la matriz de correlación de Pearson mostró relaciones lógicas, destacando la fuerte correlación positiva (0.825) entre la antigüedad y el cargo total acumulado del usuario. Esta dinámica queda en evidencia en la siguiente matriz:

![Matriz de Correlación](./resultados/clasificacion/plots/corr_pearson.png)

La limpieza final de los datos no se hizo solo con cortes estadísticos estrictos; al revisar los gráficos, decidimos dejar variables como la facturación electrónica (PaperlessBilling), ya que mostraban tendencias muy claras sobre la fuga de clientes, a pesar de que su correlación matemática fuera más baja.

#### Ingeniería de características
Para representar mejor el comportamiento del usuario, creamos un transformador personalizado (TelcoFeatureEngineer) que se integra directamente al flujo del modelo. Este código de Python generó cinco variables nuevas:

- Total_Servicios_Extra: Suma de hasta seis servicios adicionales contratados, para medir qué tan "amarrado" está el cliente a la empresa.

- Pago_Automatico: Variable binaria que indica si el cliente usa pago automático, lo que ayuda a evaluar si hay trabas o problemas en el cobro mensual.

- Vive_Solo: Medida de apoyo familiar generada al revisar si el cliente no tiene pareja ni dependientes.

- Aumento_Tarifa: Diferencia calculada entre el cargo mensual actual y el promedio histórico que ha pagado el usuario.

- Perfil_Alto_Riesgo: Variable binaria que marca a los clientes que tienen contrato mes a mes y además usan fibra óptica.

Al mismo tiempo, este paso descartó columnas que no aportaban valor (como el género o el servicio telefónico) y eliminó las variables originales que ya estaban resumidas en las nuevas, dejando el dataset más limpio.

#### Partición del conjunto de datos
La variable objetivo Churn se pasó a formato binario (1 y 0) para que los modelos la puedan procesar sin problemas. Luego, los datos se dividieron dejando un 80% para entrenar y un 20% para probar. Aquí aplicamos un muestreo estratificado (stratify=y) para asegurar que la tasa de abandono real del dataset (26.5%) se mantuviera igual en ambos grupos, evitando que el modelo aprenda de forma desbalanceada, y fijamos una semilla aleatoria (random_state=42) para poder replicar los resultados exactos en el futuro.

#### Pipelines de transformación
Para estandarizar el procesamiento, usamos un ColumnTransformer que aplica reglas específicas según el tipo de dato. A las variables categóricas nominales se les rellenaron los datos faltantes usando el valor más frecuente y se codificaron con OneHotEncoder, separando las categorías sin inventarles un orden numérico. Las variables binarias y las categóricas que sí tienen un orden lógico (como el tipo de contrato) se trabajaron con OrdinalEncoder para mantener esa estructura. Finalmente, todo esto se unió en un Pipeline final para cada modelo evaluado. Hacerlo así asegura que la creación de variables, la imputación y las transformaciones matemáticas se ejecuten en un solo bloque ordenado, evitando que se filtre información desde los datos de prueba hacia el entrenamiento.

# 8. Implementación de modelos de aprendizaje no supervisado

