/**
 * Base de Datos de Preguntas Frecuentes sobre IA
 * Sistema local sin dependencia de APIs externas
 */

export interface FAQEntry {
  id: string;
  keywords: string[];
  question: string;
  answer: string;
  relatedMilestones: string[];
}

export const FAQ_DATABASE: FAQEntry[] = [
  {
    id: "what-is-ai",
    keywords: [
      "qué es",
      "definición",
      "inteligencia artificial",
      "significado",
    ],
    question: "¿Qué es la Inteligencia Artificial?",
    answer:
      "La Inteligencia Artificial (IA) es la capacidad de las máquinas para realizar tareas que normalmente requieren inteligencia humana, como el aprendizaje, el razonamiento, la resolución de problemas y la percepción. El término fue acuñado oficialmente en 1956 durante la Conferencia de Dartmouth.",
    relatedMilestones: ["dartmouth-1956", "turing-1950"],
  },
  {
    id: "turing-test",
    keywords: [
      "test de turing",
      "turing",
      "máquinas pensantes",
      "imitation game",
    ],
    question: "¿Qué es el Test de Turing?",
    answer:
      "El Test de Turing, propuesto por Alan Turing en 1950, es un criterio para determinar si una máquina puede exhibir comportamiento inteligente indistinguible del de un ser humano. En el test, un evaluador humano mantiene conversaciones con una máquina y un humano sin saber cuál es cuál. Si el evaluador no puede distinguir la máquina del humano, se dice que la máquina ha pasado el test.",
    relatedMilestones: ["turing-1950"],
  },
  {
    id: "neural-networks",
    keywords: ["redes neuronales", "neuronas", "perceptrón", "deep learning"],
    question: "¿Qué son las redes neuronales?",
    answer:
      'Las redes neuronales son modelos computacionales inspirados en el cerebro humano. Están compuestas por capas de "neuronas" artificiales conectadas entre sí, donde cada conexión tiene un "peso" que se ajusta durante el entrenamiento. El Perceptrón (1958) fue la primera red neuronal práctica, y el algoritmo de Backpropagation (1986) permitió entrenar redes profundas, dando origen al Deep Learning moderno.',
    relatedMilestones: ["perceptron-1958", "backprop-1986", "imagenet-2012"],
  },
  {
    id: "machine-learning",
    keywords: ["aprendizaje automático", "machine learning", "ml", "entrenar"],
    question: "¿Qué es el Machine Learning?",
    answer:
      "El Machine Learning (Aprendizaje Automático) es una rama de la IA donde las máquinas aprenden patrones a partir de datos sin ser explícitamente programadas. En lugar de seguir reglas fijas, los algoritmos de ML ajustan sus parámetros internos basándose en ejemplos, mejorando su rendimiento con la experiencia.",
    relatedMilestones: ["svm-1995", "backprop-1986"],
  },
  {
    id: "deep-learning",
    keywords: ["deep learning", "aprendizaje profundo", "redes profundas"],
    question: "¿Qué es Deep Learning?",
    answer:
      "Deep Learning es un subcampo del Machine Learning que usa redes neuronales con múltiples capas (profundas) para aprender representaciones jerárquicas de datos. La revolución comenzó en 2012 con AlexNet, que demostró que las redes profundas entrenadas con GPUs podían superar dramáticamente a los métodos tradicionales en visión por computadora.",
    relatedMilestones: ["imagenet-2012", "lenet-1989", "resnet-2015"],
  },
  {
    id: "transformers",
    keywords: ["transformers", "attention", "atención", "gpt", "bert"],
    question: "¿Qué son los Transformers?",
    answer:
      'Los Transformers son una arquitectura de red neuronal introducida por Google en 2017 con el paper "Attention Is All You Need". Revolucionaron el procesamiento de lenguaje natural al usar mecanismos de "atención" que permiten al modelo enfocarse en las partes relevantes de la entrada. Son la base de modelos como GPT, BERT, y ChatGPT.',
    relatedMilestones: ["transformers-2017", "chatgpt-2022"],
  },
  {
    id: "ai-winter",
    keywords: ["invierno", "crisis", "winter", "problemas"],
    question: '¿Qué fue el "Invierno de la IA"?',
    answer:
      'Los "Inviernos de la IA" fueron períodos (principalmente en los 70s y finales de los 80s) donde el entusiasmo y la financiación en IA se desplomaron debido a promesas incumplidas y limitaciones técnicas. Las expectativas exageradas chocaron con la realidad del hardware limitado y algoritmos insuficientes. El campo revivió con avances como Backpropagation (1986) y posteriormente con Deep Learning (2012).',
    relatedMilestones: ["backprop-1986", "imagenet-2012"],
  },
  {
    id: "symbolic-ai",
    keywords: ["simbólica", "gofai", "reglas", "expertos", "lógica"],
    question: "¿Qué es la IA Simbólica?",
    answer:
      'La IA Simbólica (también llamada GOFAI - Good Old-Fashioned AI) es el enfoque clásico que representa el conocimiento mediante símbolos y reglas lógicas. Los Sistemas Expertos de los años 70 son un ejemplo: contenían bases de conocimiento con reglas del tipo "Si X entonces Y". Aunque útil para dominios específicos, este enfoque mostró limitaciones para tareas como reconocimiento de patrones.',
    relatedMilestones: ["dartmouth-1956", "expert-1970"],
  },
  {
    id: "generative-ai",
    keywords: [
      "generativa",
      "generar",
      "crear",
      "gans",
      "difusión",
      "dall-e",
      "midjourney",
    ],
    question: "¿Qué es la IA Generativa?",
    answer:
      'La IA Generativa se refiere a modelos capaces de crear contenido nuevo (imágenes, texto, audio, video) a partir de descripciones o ejemplos. Incluye tecnologías como GANs (2014), Modelos de Difusión (2022) para imágenes, y Transformers generativos como GPT para texto. Estos modelos "aprenden" las distribuciones de datos y pueden generar muestras nuevas realistas.',
    relatedMilestones: [
      "gans-2014",
      "diffusion-2022",
      "chatgpt-2022",
      "sora-2024",
    ],
  },
  {
    id: "ai-alignment",
    keywords: ["alineación", "rlhf", "seguridad", "ética", "alignment"],
    question: "¿Qué es la alineación de IA?",
    answer:
      "La alineación de IA es el problema de asegurar que los sistemas de IA actúen de acuerdo con los valores e intenciones humanas. RLHF (Reinforcement Learning from Human Feedback), usado en ChatGPT, es una técnica de alineación donde humanos califican las respuestas del modelo para que aprenda a ser útil, honesto e inofensivo.",
    relatedMilestones: ["chatgpt-2022", "reasoning-2025"],
  },
  {
    id: "how-train-model",
    keywords: ["cómo entrenar", "proceso", "pasos", "entrenar modelo"],
    question: "¿Cómo se entrena un modelo de IA?",
    answer:
      "Entrenar un modelo de IA involucra estos pasos: (1) Recopilar y preparar datos (limpiar, etiquetar), (2) Elegir una arquitectura de modelo, (3) Inicializar parámetros aleatoriamente, (4) Alimentar datos al modelo y calcular predicciones, (5) Calcular el error (loss), (6) Usar backpropagation para calcular gradientes, (7) Actualizar parámetros con un optimizador (ej: Adam), (8) Repetir con diferentes batches de datos por múltiples epochs, (9) Evaluar en datos de validación, (10) Ajustar hiperparámetros si es necesario.",
    relatedMilestones: ["backprop-1986", "imagenet-2012"],
  },
  {
    id: "gpu-vs-cpu",
    keywords: ["gpu", "cpu", "hardware", "nvidia", "tarjeta gráfica"],
    question: "¿Por qué se usan GPUs para entrenar IA?",
    answer:
      "Las GPUs (Graphics Processing Units) tienen miles de cores simples que pueden realizar muchas operaciones matemáticas en paralelo, mientras que las CPUs tienen pocos cores complejos para tareas secuenciales. El entrenamiento de redes neuronales requiere millones de multiplicaciones matriciales que son altamente paralelizables, lo que hace a las GPUs hasta 50x más rápidas que CPUs para deep learning. AlexNet en 2012 demostró el poder de las GPUs, entrenando en días lo que en CPU habría tomado meses.",
    relatedMilestones: ["imagenet-2012"],
  },
  {
    id: "supervised-unsupervised",
    keywords: ["supervisado", "no supervisado", "tipos de aprendizaje"],
    question:
      "¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?",
    answer:
      "Aprendizaje SUPERVISADO: Los datos vienen con etiquetas/respuestas correctas. El modelo aprende a mapear entradas a salidas (ej: foto → 'gato'). Ejemplos: clasificación, regresión. Aprendizaje NO SUPERVISADO: No hay etiquetas. El modelo encuentra patrones por sí solo (ej: agrupar clientes similares). Ejemplos: clustering, reducción dimensional. También existe APRENDIZAJE POR REFUERZO: el modelo aprende de recompensas/castigos (ej: enseñar a un robot a caminar).",
    relatedMilestones: ["perceptron-1958", "word2vec-2013"],
  },
  {
    id: "what-is-gpt",
    keywords: ["gpt", "generative pre-trained transformer", "openai"],
    question: "¿Qué es GPT?",
    answer:
      "GPT (Generative Pre-trained Transformer) es una familia de modelos de lenguaje desarrollados por OpenAI. Son modelos Transformer (2017) entrenados en cantidades masivas de texto de internet para predecir la siguiente palabra en una secuencia. GPT-3 (2020) tiene 175 mil millones de parámetros. GPT-3.5, combinado con RLHF, dio origen a ChatGPT (2022). GPT-4 (2023) es multimodal (texto + imágenes).",
    relatedMilestones: ["transformers-2017", "chatgpt-2022"],
  },
  {
    id: "what-is-cnn",
    keywords: ["cnn", "convolutional", "convolucional", "visión", "imágenes"],
    question: "¿Qué es una CNN?",
    answer:
      "Una CNN (Convolutional Neural Network) es una arquitectura de red neuronal especializada en procesar datos con estructura de cuadrícula, especialmente imágenes. Usa 'filtros convolucionales' que se deslizan sobre la imagen detectando características locales como bordes, texturas y patrones. Las primeras capas detectan características simples (líneas), las capas profundas combinan esas características en conceptos complejos (ojos, caras). LeNet (1989) y AlexNet (2012) son CNNs famosas.",
    relatedMilestones: ["lenet-1989", "imagenet-2012", "resnet-2015"],
  },
  {
    id: "what-is-lstm",
    keywords: ["lstm", "rnn", "recurrente", "secuencias", "memoria"],
    question: "¿Qué es una LSTM?",
    answer:
      "LSTM (Long Short-Term Memory) es un tipo de red neuronal recurrente diseñada para procesar secuencias de datos (texto, series de tiempo, audio) recordando información a largo plazo. Introducidas en 1997, las LSTMs usan 'puertas' (forget gate, input gate, output gate) para controlar qué información mantener o descartar. Fueron dominantes en NLP hasta que los Transformers (2017) las superaron en la mayoría de tareas.",
    relatedMilestones: ["lstm-1997", "transformers-2017"],
  },
  {
    id: "what-is-agi",
    keywords: ["agi", "inteligencia artificial general", "superinteligencia"],
    question: "¿Qué es AGI?",
    answer:
      "AGI (Artificial General Intelligence) o Inteligencia Artificial General se refiere a sistemas de IA que pueden entender, aprender y aplicar conocimiento en cualquier tarea intelectual que un humano pueda hacer, no solo tareas específicas. A diferencia de la IA 'estrecha' actual (especializada en una cosa), una AGI sería flexible y generalizable. Aún no existe verdadera AGI; es un objetivo de investigación a largo plazo con profundas implicaciones éticas y de seguridad.",
    relatedMilestones: ["reasoning-2025"],
  },
  {
    id: "what-is-reinforcement-learning",
    keywords: ["reinforcement learning", "refuerzo", "recompensa", "agente"],
    question: "¿Qué es Reinforcement Learning?",
    answer:
      "Reinforcement Learning (Aprendizaje por Refuerzo) es un paradigma donde un 'agente' aprende a tomar decisiones interactuando con un 'ambiente'. El agente realiza acciones, recibe recompensas (positivas o negativas), y aprende una 'política' que maximiza la recompensa acumulada. Es como entrenar a un perro con premios. Ejemplos famosos: AlphaGo derrotando al campeón de Go, robots aprendiendo a caminar, ChatGPT usando RLHF.",
    relatedMilestones: ["chatgpt-2022", "deepblue-1997"],
  },
  {
    id: "what-is-embedding",
    keywords: ["embedding", "word2vec", "representación vectorial"],
    question: "¿Qué son los embeddings?",
    answer:
      "Un embedding es una representación vectorial (numérica) de datos (palabras, imágenes, usuarios) que captura su significado o características en un espacio de menor dimensión. Word2Vec (2013) revolucionó NLP al crear embeddings donde palabras similares tienen vectores cercanos, permitiendo operaciones como 'Rey - Hombre + Mujer = Reina'. Los Transformers modernos crean embeddings contextuales donde la misma palabra puede tener diferentes vectores según el contexto.",
    relatedMilestones: ["word2vec-2013", "transformers-2017"],
  },
  {
    id: "what-is-attention",
    keywords: ["attention", "atención", "self-attention", "mecanismo"],
    question: "¿Qué es el mecanismo de atención?",
    answer:
      "El mecanismo de atención permite a los modelos 'enfocarse' en las partes más relevantes de la entrada al procesar información. En lugar de tratar todas las palabras por igual, la atención calcula scores de importancia. Self-attention (usado en Transformers) permite que cada palabra 'mire' a todas las demás para entender el contexto. Ejemplo: en 'El animal no cruzó la calle porque estaba cansado', la atención ayuda a entender que 'estaba' se refiere a 'animal'.",
    relatedMilestones: ["transformers-2017", "chatgpt-2022"],
  },
  {
    id: "what-is-fine-tuning",
    keywords: ["fine-tuning", "ajuste fino", "transfer learning", "adaptar"],
    question: "¿Qué es fine-tuning?",
    answer:
      "Fine-tuning es el proceso de tomar un modelo pre-entrenado en una tarea general y re-entrenarlo en datos específicos para una tarea particular. Es mucho más eficiente que entrenar desde cero. Por ejemplo, tomar BERT (pre-entrenado en Wikipedia) y fine-tunearlo con reseñas de películas para clasificar sentimientos. ChatGPT es GPT-3.5 fine-tuned con RLHF para ser conversacional.",
    relatedMilestones: ["chatgpt-2022", "transformers-2017"],
  },
  {
    id: "what-is-gradient-descent",
    keywords: ["gradient descent", "optimización", "descenso de gradiente"],
    question: "¿Qué es gradient descent?",
    answer:
      "Gradient Descent (Descenso de Gradiente) es el algoritmo de optimización fundamental en ML. Imagina estar en una montaña del error en la niebla. El gradiente te dice en qué dirección es más empinada la pendiente. Para minimizar el error, caminas en dirección opuesta al gradiente (cuesta abajo). Repites este proceso iterativamente: calcular gradiente → actualizar parámetros → recalcular, hasta converger a un mínimo. Variantes: SGD (Stochastic), Adam,  RMSprop.",
    relatedMilestones: ["backprop-1986"],
  },
  {
    id: "what-is-batch-size",
    keywords: ["batch", "batch size", "minibatch", "tamaño de lote"],
    question: "¿Qué es batch size?",
    answer:
      "El batch size es el número de ejemplos de entrenamiento procesados antes de actualizar los parámetros del modelo. En lugar de procesar todos los datos de una vez (batch completo) o un ejemplo a la vez (SGD), se usan 'mini-batches' (ej: 32, 64, 256 ejemplos). Batch size pequeño: actualizaciones frecuentes pero ruidosas. Batch size grande: actualizaciones estables pero requiere más memoria. Es un hiperparámetro importante.",
    relatedMilestones: ["backprop-1986", "imagenet-2012"],
  },
  {
    id: "what-is-learning-rate",
    keywords: ["learning rate", "tasa de aprendizaje", "lr", "hiperparámetro"],
    question: "¿Qué es el learning rate?",
    answer:
      "El learning rate (tasa de aprendizaje) controla qué tan grandes son los pasos al actualizar parámetros durante el entrenamiento. Learning rate alto: convergencia rápida pero puede sobrepasar el mínimo y nunca converger. Learning rate bajo: convergencia estable pero muy lenta, puede quedarse atascado. Típicamente se usa entre 0.001 y 0.1. Técnicas modernas usan 'learning rate scheduling' (reducir gradualmente) o adaptive optimizers como Adam que ajustan automáticamente.",
    relatedMilestones: ["backprop-1986"],
  },
  {
    id: "what-is-dropout",
    keywords: ["dropout", "regularización", "overfitting prevención"],
    question: "¿Qué es dropout?",
    answer:
      "Dropout es una técnica de regularización que previene overfitting 'apagando' aleatoriamente neuronas durante el entrenamiento con cierta probabilidad (ej: 50%). Esto evita que el modelo dependa demasiado de neuronas específicas, forzándolo a aprender representaciones más robustas. Es como estudiar con diferentes grupos de amigos en lugar de solo uno: si un amigo falta, igual puedes estudiar. Durante inferencia, se usan todas las neuronas. Introducido con AlexNet (2012).",
    relatedMilestones: ["imagenet-2012"],
  },
  {
    id: "what-is-diffusion-model",
    keywords: ["diffusion", "difusión", "stable diffusion", "dall-e 2"],
    question: "¿Qué son los modelos de difusión?",
    answer:
      "Los modelos de difusión son una familia de modelos generativos que aprenden a 'limpiar' ruido gradualmente para crear imágenes. El proceso: (1) Entrenamiento: Agregan ruido gradual a imágenes reales y aprenden a revertir el proceso. (2) Generación: Empiezan con ruido puro y lo 'limpian' paso a paso hasta obtener una imagen coherente. DALL-E 2 (2022) y Stable Diffusion usan difusión, superando a GANs en calidad y diversidad de imágenes generadas.",
    relatedMilestones: ["diffusion-2022", "sora-2024"],
  },
];
