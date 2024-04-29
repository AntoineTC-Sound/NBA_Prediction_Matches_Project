# NBA Game Prediction Project
## Visión General del Proyecto
### Motivación
La motivación principal detrás de este proyecto es el desafío de predecir resultados en deportes, particularmente en partidos de la NBA, que son altamente impredecibles y volátiles. Con una pasión personal por la NBA y una gran cantidad de datos estadísticos disponibles, este proyecto tiene como objetivo aprovechar modelos estadísticos avanzados para predecir los resultados de los juegos y potencialmente extender las capacidades del modelo para predicciones durante los juegos.

### Fuente de Datos
Obtenemos nuestros datos de:
- Basketball Reference: Métricas clave y avanzadas.
- Stathead Basketball: Estadísticas avanzadas adicionales.

### Manejo de Datos
- Base de Datos en la Nube: Utilizamos Amazon Lightsail para el alojamiento de la base de datos, lo que permite el acceso remoto.
- Gestión de la Base de Datos: Se utiliza MySQL para manejar conexiones e interacciones entre diversas tablas de datos.

### Detalles de los Datos
- Datos Brutos: Se recopilaron estadísticas de todos los juegos desde la temporada 2019 en adelante, enfocándonos en los dos jugadores principales por minutos jugados de cada equipo y las estadísticas generales del equipo.

- Datos Filtrados: Optamos por excluir la temporada 2019/2020 debido a irregularidades causadas por COVID-19, centrando nuestro análisis desde la temporada 2020/2021 hasta la 2022/2023.
Preparación y Tratamiento Inicial de los Datos

### Fusión de Datos
- Game ID: Creado para vincular los conjuntos de datos de jugadores y equipos.

- Enriquecimiento de Datos: Incluido un historial de rachas ganadoras para mejorar nuestro conjunto de datos.

- Funciones Temporales: Utilizadas para gestionar y ajustar las líneas de tiempo de los datos y crear estadísticas acumulativas sobre los últimos cinco juegos sin que los resultados sean influenciados por los resultados inmediatos anteriores.

### Preprocesamiento
Dividir los datos por equipos locales y visitantes para mejorar la estructura del conjunto de datos, centrándonos en predecir el resultado desde la perspectiva del equipo local.

### Ingeniería de Características
Variables Clave
eFG% (Porcentaje de Tiros de Campo Efectivos)
ORtg (Rating Ofensivo)
Plus-Minus: Mide la diferencia de puntos cuando un jugador está en la cancha.

### Análisis Exploratorio de Datos (EDA)
Se realizaron análisis de correlación, evaluaciones de variables y análisis de componentes principales (PCA) para identificar y agrupar dimensiones clave de los datos.

### Construcción del Modelo
Técnicas Empleadas:
- Regresión Logística.
- Gaussian Bayes.
- Random Forest.
- Redes Neuronales.
- Boosting (con un modelo de votación final combinando los resultados de boosting y redes neuronales).

### Rendimiento
Nuestro modelo conjunto alcanzó una precisión del 70%, indicando una capacidad robusta para predecir correctamente los resultados de los partidos de la NBA en el 70% de los casos.

### Información desde Power BI
- Mejores Precisones: Identificados jugadores cuya presencia correlaciona positivamente con puntuaciones más altas del equipo.
- Peores Precisones: Analizados equipos y jugadores que contribuyen de manera menos predecible a los resultados de los juegos.
- Análisis de Impacto Ofensivo y del Jugador: Explorado la interacción entre las estrategias de equipo y las actuaciones individuales de los jugadores.

### Conclusión
Aunque predecir los resultados de los partidos de la NBA sigue siendo un desafío debido a su naturaleza impredecible, nuestro modelo demuestra un poder predictivo sustancial. Las expansiones futuras podrían incluir predicciones más detalladas como totales de puntos y eficiencias de tiro. El modelo actual establece una base sólida para una exploración y refinamiento adicionales.
