# Informe Técnico: Implementación de Modelos Predictivos de Churn en Telecomunicaciones

- Equipo de Trabajo: Felipe Huincaman, David  Ramírez Luna y Octavio Chávez
- Asignatura: Machine Learning
- Docente: Profesora Jazna Meza Hidalgo
- Fecha de elaboración:

# 1. Problema de negocio
Para las empresas de telecomunicaciones, la retención de clientes es un desafío empresarial crítico. En un mercado donde los usuarios dependen de la conectividad diaria y pueden migrar fácilmente hacia la competencia ante la más mínima interrupción, mala experiencia o una oferta más atractiva, la tasa de abandono (churn) representa uno de los mayores riesgos financieros del sector. Considerando que adquirir un nuevo usuario cuesta hasta cinco veces más que retener a uno actual, no controlar el abandono se traduce en una pérdida directa de ingresos que amenaza la rentabilidad de la compañía.

Para mitigar este impacto, la empresa necesita dejar de operar de manera reactiva y adoptar una gestión preventiva e integral del ciclo de vida del cliente. El principal desafío analítico para el negocio requiere responder a tres preguntas estratégicas fundamentales: ¿Cómo se agrupan nuestros clientes según sus hábitos de consumo y facturación?, ¿Cuáles de ellos están en riesgo inminente de cancelar su servicio?, y ¿Cuánto tiempo de permanencia esperado le queda a cada usuario en la compañía?

Al comprender la diversidad de perfiles mediante la segmentación de la cartera, anticipar de forma precisa las bajas y proyectar el tiempo de vida del cliente, la empresa de telecomunicaciones puede optimizar sus presupuestos de retención. Esto permite a los equipos comerciales priorizar esfuerzos en los usuarios más rentables, diseñar campañas de fidelización personalizadas a las necesidades de cada nicho y mejorar proactivamente la experiencia del servicio, convirtiendo la retención en un pilar estratégico de crecimiento sostenible.

# 2. Objetivos del proyecto
#### Objetivo general del proyecto:
- Diseñar una estrategia integral basada en datos para mitigar el impacto financiero del abandono de clientes en la empresa de telecomunicaciones, sustentada en la segmentación de los usuarios según sus patrones de comportamiento y consumo, la detección anticipada de futuras cancelaciones y la estimación del tiempo de permanencia esperado para cada cliente, con el fin de diseñar campañas de fidelización personalizadas y optimizar la inversión del equipo de retención.

#### Objetivos específicos de etapa de Implementación de Modelos de Machine Learning:
1. Comprensión de la base de clientes: Analizar el comportamiento histórico y las características de los usuarios para identificar los factores clave y los motivos que impulsan las cancelaciones de servicio, asegurando información confiable para la toma de decisiones.
2. Segmentación del mercado: Descubrir y definir perfiles de clientes con comportamientos de consumo y facturación similares, permitiendo a los equipos de marketing diseñar campañas de fidelización específicas y adaptadas a las necesidades de cada arquetipo.
3. Detección de fugas: Identificar de forma anticipada a los clientes que presentan una alta probabilidad de abandonar la compañía, proporcionando al negocio alertas tempranas para intervenir de manera proactiva antes de que se concrete la baja.
4. Proyección de valor y permanencia: Estimar el tiempo de permanencia esperado de los usuarios activos, con el fin de priorizar los presupuestos de retención hacia aquellos clientes que representan una mayor rentabilidad a largo plazo para la empresa.
5. Validación de confiabilidad de la solución: Asegurar la precisión y fiabilidad de las herramientas predictivas desarrolladas, garantizando que las estrategias comerciales y las inversiones en retención se basen en proyecciones certeras que minimicen los costos por falsas alarmas o fugas no detectadas.
6. Garantía de responsabilidad ética: Evaluar el impacto comercial y social de las estrategias propuestas, garantizando el respeto a la privacidad de los clientes, la transparencia en el uso de su información y la ausencia de tratos discriminatorios o sesgados en las campañas de fidelización.

# 3. KPIs que resolverán el problema de negocio


# 4. Descripción General y Calidad del Conjunto de Datos
El conjunto de datos contiene información sobre los clientes de una empresa de telecomunicaciones y si se dieron de baja (cancelaron su servicio) o no. Cada fila representa a un cliente, cada columna contiene los atributos del cliente descritos. El conjunto de datos original está compuesto por 7043 registros y 21 características.

#### El conjunto de datos incluye información sobre:

* Clientes que se dieron de baja en el último mes – la columna se llama Churn (tasa de abandono).
* Servicios a los que cada cliente se ha suscrito – teléfono, múltiples líneas, internet, seguridad en línea, respaldo en línea, protección de dispositivos, soporte técnico y streaming de TV y películas.
* Información de la cuenta del cliente – cuánto tiempo llevan como clientes, contrato, método de pago, facturación electrónica, cargos mensuales y cargos totales.
* Información demográfica sobre los clientes – género, rango de edad, y si tienen pareja y dependientes.

#### Calidad de Datos:

* Variables Categóricas: Alta presencia de datos cualitativos estructurados como texto.
* Duplicidad: No se han detectado registros duplicados en el conjunto de datos.
* Completitud: La integridad de los datos es excelente. El único hallazgo de valores nulos se
presentó en la variable TotalCharges (11 registros faltantes), los cuales corresponden
estrictamente a clientes con una antiguedad (tenure) de 0 meses.
* Análisis de Valores Atípicos (Outliers): Se evaluaron las variables numéricas continuas
mediante el método de Rango Intercuartil (IQR). Los resultados indicaron una ausencia
total de valores atípicos en las características tenure, MonthlyCharges y
TotalCharges.

# 5. Análisis exploratorio de los datos (EDA)

# 6. Metodología utilizada (CRISP-DM).
Para el desarrollo de este proyecto, se adoptó la metodología **CRISP-DM**, garantizando un enfoque estructurado desde la comprensión del problema de negocio hasta la preparación para el despliegue del modelo.

### 1. Comprensión del Negocio (Business Understanding)
El objetivo principal es predecir la tasa de abandono de clientes (Churn) en una empresa de telecomunicaciones para optimizar las estrategias de retención. Se identificó el impacto financiero de los errores predictivos: los **Falsos Negativos** implican pérdida directa de clientes e ingresos, mientras que los **Falsos Positivos** generan ineficiencia en el presupuesto de marketing. Adicionalmente, se integraron los requisitos éticos y normativos para garantizar la privacidad y el resguardo de datos sensibles desde la concepción del proyecto.

### 2. Comprensión de los Datos (Data Understanding)
Se realizó un Análisis Exploratorio de Datos (EDA) para identificar los patrones que explican el abandono. Se utilizó la métrica **V de Cramér** para evaluar características categóricas y correlación de Pearson para las numéricas. La selección de variables combinó rigor matemático con evidencia empírica visual; reteniendo variables de menor puntuación matemática debido a que sus distribuciones gráficas demuestran una clara influencia en el comportamiento del usuario.

### 3. Preparación de los Datos (Data Preparation)
El procesamiento se estructuró mediante pipelines para evitar fugas de información y asegurar consistencia técnica y legal:
* **Privacidad y Gobernanza:** Se aplicó seudonimización (Hashing criptográfico) al identificador "CustomerID" y minimización de datos, descartando columnas sin peso predictivo justificado.
* **Limpieza e Imputación:** Automatización del tratamiento de valores nulos e inconsistencias detectadas.
* **Codificación de Variables:** Implementación de One-Hot Encoding para variables nominales policategóricas, optimizando la interpretación matemática del algoritmo.

### 4. Modelado (Modeling)
Para mitigar el sesgo introducido por el desbalance de clases, se determinó la aplicación de técnicas de re-muestreo exclusivamente sobre el conjunto de entrenamiento. Se sugirió la técnica **SMOTE** para generar registros sintéticos mediante vecinos cercanos para evitar la duplicidad exacta de datos, permitiendo al algoritmo trazar fronteras de decisión mucho más precisas y equitativas.

### 5. Evaluación (Evaluation)
La validación del rendimiento no se basa únicamente en la exactitud global, sino en la capacidad del modelo para equilibrar la detección de la clase minoritaria sin disparar los Falsos Positivos. Se priorizan métricas robustas frente a datos desbalanceados para asegurar que las alertas tempranas de abandono sean financieramente viables para la compañía.

### 6. Despliegue (Deployment)
La arquitectura del proyecto contempla las normativas vigentes para su paso a producción. Esto incluye la necesidad operativa de implementar un Control de Acceso Basado en Roles    para restringir el uso de los datos productivos solo al personal autorizado bajo el principio del menor privilegio, además de garantizar los mecanismos necesarios para la transparencia algorítmica y la atención a los derechos ARCO de los titulares.