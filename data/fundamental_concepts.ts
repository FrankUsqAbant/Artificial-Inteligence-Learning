/**
 * Conceptos Fundamentales de Inteligencia Artificial
 * Glosario de términos básicos para principiantes
 */

export interface FundamentalConcept {
  id: string;
  term: string;
  shortDefinition: string;
  detailedExplanation: string;
  analogy: string;
  example?: string;
  relatedConcepts: string[];
  difficulty: "beginner" | "intermediate" | "advanced";
}

export const FUNDAMENTAL_CONCEPTS: Record<string, FundamentalConcept> = {
  "inteligencia-artificial": {
    id: "inteligencia-artificial",
    term: "Inteligencia Artificial (IA)",
    shortDefinition:
      "Capacidad de las máquinas para realizar tareas que normalmente requieren inteligencia humana.",
    detailedExplanation: `La Inteligencia Artificial es un campo de la informática que busca crear sistemas capaces de realizar tareas que, cuando las hacen humanos, requieren inteligencia: aprender de experiencias, reconocer patrones, entender lenguaje, tomar decisiones, resolver problemas complejos.

No se trata de crear "conciencia" o "sentimientos", sino de construir algoritmos que puedan encontrar patrones en datos y usar esos patrones para hacer predicciones o tomar decisiones útiles.`,
    analogy:
      "Como enseñar a un niño a reconocer animales mostrándole muchas fotos, la IA aprende patrones de ejemplos.",
    example:
      "Cuando Netflix te recomienda series, usa IA para analizar qué viste antes y predecir qué te gustará.",
    relatedConcepts: ["machine-learning", "modelo", "algoritmo"],
    difficulty: "beginner",
  },

  "machine-learning": {
    id: "machine-learning",
    term: "Machine Learning (Aprendizaje Automático)",
    shortDefinition:
      "Rama de IA donde los algoritmos aprenden patrones de datos sin ser programados explícitamente.",
    detailedExplanation: `En programación tradicional, escribes reglas específicas: "Si X entonces Y". En Machine Learning, muestras ejemplos al algoritmo y él descubre las reglas por sí mismo.

Por ejemplo, en lugar de programar manualmente todas las reglas para detectar spam (contiene "gratis", tiene muchos signos de exclamación, etc.), le muestras al algoritmo miles de emails marcados como spam/no-spam, y él aprende qué patrones indican spam.

El algoritmo ajusta sus parámetros internos (pesos) para minimizar errores, mejorando con cada ejemplo.`,
    analogy:
      "Como aprender a andar en bicicleta: nadie te da una lista de reglas, practicas y tu cerebro ajusta automáticamente el balance.",
    example:
      "Gmail usa ML para filtrar spam. No tiene reglas escritas, aprendió de millones de emails qué es spam.",
    relatedConcepts: ["entrenamiento", "modelo", "datos", "supervision"],
    difficulty: "beginner",
  },

  modelo: {
    id: "modelo",
    term: "Modelo",
    shortDefinition:
      "Representación matemática que aprende patrones de datos y hace predicciones.",
    detailedExplanation: `Un modelo es como una función matemática compleja con muchos parámetros ajustables. Recibe entradas (ej: una imagen) y produce salidas (ej: "es un gato").

Durante el entrenamiento, el modelo ajusta sus parámetros internos (pesos) para mapear correctamente entradas a salidas. Una vez entrenado, puedes usar el modelo para hacer predicciones en datos nuevos nunca vistos.

Tipos comunes: redes neuronales, árboles de decisión, regresión lineal, etc.`,
    analogy:
      "Como una receta de cocina con ingredientes ajustables. Pruebas diferentes cantidades hasta que el resultado sea perfecto.",
    example:
      "GPT es un modelo con 175 mil millones de parámetros que mapea texto de entrada a texto de salida.",
    relatedConcepts: [
      "parametros",
      "entrenamiento",
      "prediccion",
      "arquitectura",
    ],
    difficulty: "beginner",
  },

  entrenamiento: {
    id: "entrenamiento",
    term: "Entrenamiento",
    shortDefinition:
      "Proceso de ajustar los parámetros del modelo usando datos de ejemplo.",
    detailedExplanation: `El entrenamiento es el proceso donde el modelo "aprende". Se le muestran ejemplos (datos de entrenamiento), el modelo hace predicciones, se calcula qué tan erradas son, y se ajustan los parámetros para reducir el error.

Este proceso se repite miles o millones de veces (epochs) hasta que el modelo alcanza buen rendimiento.

Fórmula básica:
1. Modelo hace predicción
2. Se calcula error (loss)
3. Se calculan gradientes (cuánto cambiar cada parámetro)
4. Se actualizan parámetros
5. Repetir`,
    analogy:
      "Como estudiar para un examen: lees (datos), resuelves ejercicios (predicciones), ves tus errores, y ajustas tu entendimiento.",
    example:
      "Entrenar un clasificador de imágenes: mostrarle 1 millón de fotos etiquetadas hasta que aprenda a distinguir gatos de perros.",
    relatedConcepts: [
      "datos",
      "loss",
      "gradiente",
      "epochs",
      "backpropagation",
    ],
    difficulty: "beginner",
  },

  parametros: {
    id: "parametros",
    term: "Parámetros (Pesos)",
    shortDefinition:
      "Valores numéricos ajustables dentro del modelo que se optimizan durante el entrenamiento.",
    detailedExplanation: `Los parámetros son los "botones de ajuste" del modelo. En redes neuronales, son los pesos de las conexiones entre neuronas.

Cuando dices que GPT-3 tiene "175 mil millones de parámetros", significa que tiene 175 mil millones de números que fueron ajustados durante el entrenamiento para mapear correctamente entradas a salidas.

Más parámetros = mayor capacidad de aprender patrones complejos, pero también más datos y computación necesarios.`,
    analogy:
      "Como perillas de un ecualizador de audio. Cada perilla (parámetro) controla una frecuencia, y ajustas todas para obtener el mejor sonido.",
    example:
      "Una red neuronal simple puede tener 10,000 parámetros. GPT-4 tiene cientos de miles de millones.",
    relatedConcepts: ["modelo", "entrenamiento", "optimizacion"],
    difficulty: "intermediate",
  },

  datos: {
    id: "datos",
    term: "Datos de Entrenamiento",
    shortDefinition: "Ejemplos usados para enseñar al modelo patrones.",
    detailedExplanation: `Los datos son el combustible del machine learning. Sin buenos datos, no hay aprendizaje.

Datos típicamente consisten en:
- **Entradas (X)**: Lo que el modelo recibe (imágenes, texto, números)
- **Etiquetas (Y)**: La respuesta correcta (ej: "gato", "spam", "5 estrellas")

Calidad > Cantidad (aunque cantidad también importa):
- Datos limpios, diversos, representativos
- Sin sesgos sistemáticos
- Bien etiquetados

División típica:
- 70-80% entrenamiento
- 10-15% validación
- 10-15% prueba (test)`,
    analogy:
      "Como flashcards para estudiar. Las preguntas son las entradas (X), las respuestas son las etiquetas (Y).",
    example:
      "ImageNet: 14 millones de imágenes etiquetadas a mano. Usado para entrenar la mayoría de modelos de visión.",
    relatedConcepts: ["entrenamiento", "supervision", "etiquetas", "dataset"],
    difficulty: "beginner",
  },

  supervision: {
    id: "supervision",
    term: "Aprendizaje Supervisado",
    shortDefinition:
      "Tipo de ML donde el modelo aprende de ejemplos con sus respuestas correctas.",
    detailedExplanation: `En aprendizaje supervisado, cada ejemplo de entrenamiento viene con su "etiqueta" o respuesta correcta. El modelo aprende a mapear entradas a salidas viendo estos pares.

Ejemplo: Para entrenar un clasificador de emails como spam/no-spam:
- Entrada: Texto del email
- Etiqueta: "Spam" o "No spam"

El modelo aprende qué características del texto predicen cada categoría.

Otros tipos:
- **No supervisado**: Sin etiquetas, el modelo encuentra patrones por sí solo
- **Por refuerzo**: El modelo aprende de recompensas/castigos`,
    analogy:
      "Como aprender matemáticas con un libro de respuestas. Resuelves ejercicios y verificas si acertaste.",
    example:
      'Reconocimiento de dígitos manuscritos: Mostrar imágenes de "3" etiquetadas como "3" hasta que el modelo aprenda.',
    relatedConcepts: ["datos", "etiquetas", "clasificacion", "regresion"],
    difficulty: "intermediate",
  },

  overfitting: {
    id: "overfitting",
    term: "Overfitting (Sobreajuste)",
    shortDefinition:
      "Cuando el modelo memoriza los datos de entrenamiento en lugar de aprender patrones generales.",
    detailedExplanation: `Overfitting ocurre cuando el modelo se ajusta demasiado a los datos de entrenamiento, memorizando incluso el ruido y peculiaridades específicas, en lugar de aprender patrones generalizables.

Síntomas:
- Alta precisión en datos de entrenamiento
- Baja precisión en datos nuevos (test)

Es como estudiar memorizando ejercicios específicos en lugar de entender conceptos. Funciona en el examen de práctica (datos de entrenamiento), fallas en el examen real (datos de test).

Soluciones:
- Más datos de entrenamiento
- Regularización (penalizar modelos complejos)
- Dropout (apagar neuronas aleatoriamente)
- Early stopping (parar entrenamiento antes)`,
    analogy:
      "Como memorizar respuestas de exámenes anteriores palabra por palabra. Funciona si el examen es idéntico, pero fallas si cambian las preguntas.",
    example:
      "Un modelo que memoriza cada imagen de entrenamiento perfectamente pero no reconoce nuevas fotos similares.",
    relatedConcepts: [
      "underfitting",
      "generalizacion",
      "regularizacion",
      "validacion",
    ],
    difficulty: "intermediate",
  },

  loss: {
    id: "loss",
    term: "Loss (Función de Pérdida)",
    shortDefinition:
      "Métrica que mide qué tan equivocadas son las predicciones del modelo.",
    detailedExplanation: `La función de loss cuantifica el error del modelo. Compara las predicciones del modelo con las respuestas correctas y asigna un número: mientras mayor el loss, peor el modelo.

Durante entrenamiento, el objetivo es minimizar el loss ajustando los parámetros.

Funciones comunes:
- **MSE (Mean Squared Error)**: Para regresión, penaliza errores grandes
- **Cross-Entropy**: Para clasificación, mide diferencia entre distribuciones
- **Binary Cross-Entropy**: Para clasificación binaria (sí/no)

Fórmula MSE simple:
Loss = (1/n) × Σ(predicción - real)²`,
    analogy:
      'Como un puntaje de error en un examen. Mientras más preguntas falles, mayor tu "loss".',
    example:
      "Si el modelo predice que una casa cuesta $300k pero en realidad cuesta $350k, el loss captura ese error de $50k.",
    relatedConcepts: [
      "entrenamiento",
      "optimizacion",
      "gradiente",
      "backpropagation",
    ],
    difficulty: "intermediate",
  },

  gradiente: {
    id: "gradiente",
    term: "Gradiente",
    shortDefinition:
      "Vector que indica la dirección y magnitud del cambio necesario en cada parámetro para reducir el loss.",
    detailedExplanation: `El gradiente es un concepto matemático (derivada parcial) que indica "cómo cambiar cada parámetro para reducir el loss".

Imagina estar en una montaña del error (landscape del loss). El gradiente apunta en la dirección de máxima pendiente ascendente. Para minimizar, vas en dirección opuesta (gradiente descendente).

Para cada parámetro:
- Gradiente positivo → reducir parámetro
- Gradiente negativo → aumentar parámetro
- Gradiente cercano a cero → parámetro ya está bien

El algoritmo de backpropagation calcula eficientemente todos los gradientes de una red neuronal.`,
    analogy:
      "Como estar en una colina con niebla. El gradiente te dice en qué dirección es más empinado (subes o bajas más rápido).",
    example:
      "Si gradiente de un peso es -0.5, aumentar ese peso en 0.1 reducirá el loss aproximadamente en 0.05.",
    relatedConcepts: [
      "backpropagation",
      "optimizacion",
      "learning-rate",
      "loss",
    ],
    difficulty: "advanced",
  },

  epochs: {
    id: "epochs",
    term: "Epochs (Épocas)",
    shortDefinition:
      "Una pasada completa del modelo por todos los datos de entrenamiento.",
    detailedExplanation: `Un epoch significa que el modelo ha visto cada ejemplo de entrenamiento exactamente una vez.

Entrenar por múltiples epochs permite al modelo refinar su entendimiento:
- Epoch 1: Aprende patrones básicos
- Epoch 10: Refina y ajusta detalles
- Epoch 50: Ha visto los datos 50 veces, aprendizaje profundo

Demasiados epochs → overfitting (memoriza datos)
Muy pocos epochs → underfitting (no aprende suficiente)

Típico: 10-100 epochs, dependiendo del problema.`,
    analogy:
      "Como leer un libro de estudio completo. Cada vez que lo lees (epoch), entiendes mejor los conceptos.",
    example:
      "Entrenar un clasificador por 20 epochs significa que vio todo el dataset 20 veces completas.",
    relatedConcepts: ["entrenamiento", "batch", "iteration", "early-stopping"],
    difficulty: "beginner",
  },

  arquitectura: {
    id: "arquitectura",
    term: "Arquitectura de Red",
    shortDefinition:
      "Diseño estructural del modelo: cuántas capas, qué tipo de conexiones, etc.",
    detailedExplanation: `La arquitectura define la "forma" del modelo antes de entrenarlo. Es como el plano de un edificio antes de construirlo.

Decisiones arquitectónicas:
- **Número de capas**: Profundidad de la red
- **Número de neuronas por capa**: Ancho de la red
- **Tipo de capas**: Convolucionales, recurrentes, atención, etc.
- **Funciones de activación**: ReLU, sigmoid, tanh
- **Conexiones**: ¿Todas las capas conectadas? ¿Saltos (skip connections)?

Arquitecturas famosas:
- **ResNet**: Redes residuales con skip connections
- **Transformer**: Basada en mecanismos de atención
- **U-Net**: Para segmentación de imágenes
- **LSTM**: Para secuencias temporales`,
    analogy:
      "Como el diseño de un circuito electrónico. Defines qué componentes usar y cómo conectarlos antes de construirlo.",
    example:
      "AlexNet: 5 capas convolucionales + 3 capas fully-connected = arquitectura específica que ganó ImageNet 2012.",
    relatedConcepts: ["modelo", "capas", "neurona", "deep-learning"],
    difficulty: "intermediate",
  },

  inferencia: {
    id: "inferencia",
    term: "Inferencia (Predicción)",
    shortDefinition:
      "Usar un modelo ya entrenado para hacer predicciones en datos nuevos.",
    detailedExplanation: `Inferencia es la fase de "uso" del modelo, después del entrenamiento. Das una entrada nueva (nunca vista) y el modelo predice una salida.

Diferencias con entrenamiento:
- **No se actualizan parámetros** (están congelados)
- **Mucho más rápido** (no hay backpropagation)
- **Se hace en producción**, con datos reales

Es la fase donde el modelo genera valor real: recomienda productos, detecta tumores, traduce texto, etc.

Optimizaciones comunes:
- Cuantización (reducir precisión de pesos)
- Pruning (eliminar conexiones menos importantes)
- Distillation (crear modelo más pequeño que imita uno grande)`,
    analogy:
      "Como usar una calculadora entrenada. Ya sabes matemáticas (entrenamiento), ahora solo usas ese conocimiento para resolver nuevos problemas.",
    example:
      "Cuando usas ChatGPT, está en modo inferencia: usa sus parámetros entrenados para generar respuestas sin re-entrenar.",
    relatedConcepts: ["entrenamiento", "modelo", "produccion", "deployment"],
    difficulty: "beginner",
  },

  "transfer-learning": {
    id: "transfer-learning",
    term: "Transfer Learning",
    shortDefinition:
      "Reusar un modelo pre-entrenado en una tarea como punto de partida para otra tarea.",
    detailedExplanation: `En lugar de entrenar desde cero, usas un modelo ya entrenado en una tarea grande (ej: ImageNet) y lo ajustas (fine-tune) para tu tarea específica.

Ventajas:
- **Menos datos necesarios**: El modelo ya aprendió características básicas
- **Entrenamiento más rápido**: Solo ajustas las últimas capas
- **Mejor rendimiento**: Especialmente con pocos datos

Proceso típico:
1. Tomar modelo pre-entrenado (ej: ResNet en ImageNet)
2. Congelar capas iniciales (features genéricos)
3. Re-entrenar últimas capas con tus datos (features específicos)

Es la técnica estándar en IA moderna. Casi nadie entrena grandes modelos desde cero.`,
    analogy:
      "Como aprender francés después de saber español. No empiezas de cero, transfieres conocimiento de gramática y estructura.",
    example:
      "BERT pre-entrenado en todo internet se usa como base para clasificar sentimientos en reseñas de productos.",
    relatedConcepts: ["fine-tuning", "pre-training", "features", "embeddings"],
    difficulty: "intermediate",
  },
};

/**
 * Obtiene un concepto por ID
 */
export function getConceptById(id: string): FundamentalConcept | undefined {
  return FUNDAMENTAL_CONCEPTS[id];
}

/**
 * Obtiene conceptos por nivel de dificultad
 */
export function getConceptsByDifficulty(
  difficulty: "beginner" | "intermediate" | "advanced"
): FundamentalConcept[] {
  return Object.values(FUNDAMENTAL_CONCEPTS).filter(
    (concept) => concept.difficulty === difficulty
  );
}

/**
 * Busca conceptos por término
 */
export function searchConcepts(query: string): FundamentalConcept[] {
  const queryLower = query.toLowerCase();
  return Object.values(FUNDAMENTAL_CONCEPTS).filter(
    (concept) =>
      concept.term.toLowerCase().includes(queryLower) ||
      concept.shortDefinition.toLowerCase().includes(queryLower) ||
      concept.detailedExplanation.toLowerCase().includes(queryLower)
  );
}
