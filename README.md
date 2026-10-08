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
- Diseñar una estrategia integral basada en datos para mitigar el impacto financiero del abandono de clientes en la empresa de telecomunicaciones, sustentada en la segmentación de los usuarios según sus patrones de comportamiento y consumo, la detección anticipada de futuras cancelaciones y la estimación del tiempo de permanencia esperado para cada cliente, con el fin de diseñar campañas de fidelización personalizadas y optimizar la inversión del equipo de retención.

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

# 5. Descripción General y Calidad del Conjunto de Datos
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

# 6. Preparación y análisis exploratorio de los datos (EDA)

# 7. Implementación de modelos de aprendizaje supervisado

# 8. Implementación de modelos de aprendizaje no supervisado