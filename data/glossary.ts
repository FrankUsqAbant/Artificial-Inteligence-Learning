/**
 * Glosario de Términos de Inteligencia Artificial
 * Definiciones técnicas para referencia rápida
 */

export interface GlossaryEntry {
  term: string;
  definition: string;
  relatedMilestones: string[];
  relatedTerms: string[];
}

export const AI_GLOSSARY: Record<string, GlossaryEntry> = {
  backpropagation: {
    term: "Backpropagation (Retropropagación)",
    definition:
      "Algoritmo fundamental para entrenar redes neuronales. Calcula el gradiente del error propagándolo desde la capa de salida hacia atrás a través de la red, permitiendo ajustar los pesos de cada neurona. Publicado por Rumelhart, Hinton y Williams en 1986, resolvió el problema de entrenar redes con múltiples capas ocultas.",
    relatedMilestones: ["backprop-1986"],
    relatedTerms: ["gradient-descent", "neural-network", "deep-learning"],
  },

  "gradient-descent": {
    term: "Gradient Descent (Descenso de Gradiente)",
    definition:
      "Algoritmo de optimización que ajusta los parámetros de un modelo en la dirección opuesta al gradiente de la función de error, minimizando progresivamente el error. Es la base matemática del entrenamiento de redes neuronales.",
    relatedMilestones: ["backprop-1986", "perceptron-1958"],
    relatedTerms: ["backpropagation", "optimization"],
  },

  "neural-network": {
    term: "Red Neuronal (Neural Network)",
    definition:
      "Modelo computacional compuesto por capas de neuronas artificiales interconectadas. Cada neurona realiza una suma ponderada de sus entradas seguida de una función de activación. Las redes pueden tener arquitecturas feedforward (hacia adelante) o recurrentes (con ciclos).",
    relatedMilestones: ["perceptron-1958", "backprop-1986"],
    relatedTerms: ["perceptron", "deep-learning", "activation-function"],
  },

  perceptron: {
    term: "Perceptrón",
    definition:
      "La neurona artificial más simple, inventada por Frank Rosenblatt en 1958. Realiza una suma ponderada de sus entradas y aplica una función escalón. Puede aprender a clasificar patrones linealmente separables mediante el ajuste de pesos.",
    relatedMilestones: ["perceptron-1958"],
    relatedTerms: ["neural-network", "linear-classifier"],
  },

  "deep-learning": {
    term: "Deep Learning (Aprendizaje Profundo)",
    definition:
      "Rama del Machine Learning que usa redes neuronales con muchas capas (profundas) para aprender representaciones jerárquicas de datos. Cada capa aprende características cada vez más abstractas. Revolucionó la IA a partir de 2012 con el uso masivo de GPUs.",
    relatedMilestones: ["imagenet-2012", "backprop-1986", "lenet-1989"],
    relatedTerms: ["neural-network", "cnn", "representation-learning"],
  },

  cnn: {
    term: "CNN (Convolutional Neural Network)",
    definition:
      "Red neuronal convolucional diseñada para procesar datos con estructura tipo rejilla (como imágenes). Usa capas convolucionales que detectan patrones locales y capas de pooling que reducen dimensionalidad. LeNet-5 (1989) fue la primera CNN exitosa.",
    relatedMilestones: ["lenet-1989", "imagenet-2012", "resnet-2015"],
    relatedTerms: ["deep-learning", "convolution", "computer-vision"],
  },

  transformer: {
    term: "Transformer",
    definition:
      'Arquitectura de red neuronal basada en mecanismos de atención, introducida en 2017. Procesa secuencias en paralelo (no recurrentemente) y usa "self-attention" para ponderar la importancia de diferentes partes de la entrada. Es la base de GPT, BERT, y modelos generativos modernos.',
    relatedMilestones: ["transformers-2017", "chatgpt-2022"],
    relatedTerms: ["attention", "self-attention", "llm"],
  },

  attention: {
    term: "Attention (Atención)",
    definition:
      'Mecanismo que permite a una red neuronal enfocarse dinámicamente en partes relevantes de la entrada. Calcula scores de importancia entre elementos, permitiendo al modelo "atender" a información contextual específica. Es el componente clave de los Transformers.',
    relatedMilestones: ["transformers-2017"],
    relatedTerms: ["transformer", "self-attention"],
  },

  llm: {
    term: "LLM (Large Language Model)",
    definition:
      "Modelo de lenguaje de gran escala entrenado en enormes cantidades de texto. Aprende patrones estadísticos del lenguaje y puede generar, completar o responder texto de manera coherente. GPT-3, GPT-4 y Gemini son ejemplos de LLMs con billones de parámetros.",
    relatedMilestones: ["transformers-2017", "chatgpt-2022"],
    relatedTerms: ["transformer", "generative-ai", "rlhf"],
  },

  rlhf: {
    term: "RLHF (Reinforcement Learning from Human Feedback)",
    definition:
      "Técnica de alineación donde se entrena un modelo usando feedback humano como señal de recompensa. Humanos califican múltiples respuestas del modelo, se entrena un modelo de recompensa, y luego se optimiza el modelo original usando reinforcement learning. Usado en ChatGPT para hacerlo más útil y seguro.",
    relatedMilestones: ["chatgpt-2022"],
    relatedTerms: ["alignment", "reinforcement-learning", "llm"],
  },

  gan: {
    term: "GAN (Generative Adversarial Network)",
    definition:
      "Arquitectura donde dos redes neuronales compiten: un Generador crea datos sintéticos y un Discriminador intenta distinguir datos reales de falsos. El Generador mejora hasta crear datos indistinguibles de los reales. Inventado por Ian Goodfellow en 2014, revolucionó la generación de imágenes realistas.",
    relatedMilestones: ["gans-2014"],
    relatedTerms: ["generative-ai", "adversarial-training"],
  },

  embedding: {
    term: "Embedding (Incrustación)",
    definition:
      "Representación vectorial densa de datos discretos (palabras, tokens, entidades) en un espacio continuo de menor dimensión. Los embeddings capturan similitudes semánticas: palabras similares tienen vectores cercanos. Word2Vec (2013) popularizó los embeddings de palabras.",
    relatedMilestones: ["word2vec-2013"],
    relatedTerms: ["word2vec", "representation-learning", "vector-space"],
  },

  lstm: {
    term: "LSTM (Long Short-Term Memory)",
    definition:
      'Tipo de red neuronal recurrente diseñada para recordar información durante largos períodos. Usa "puertas" (gates) que controlan qué información olvidar, actualizar o emitir. Inventado en 1997, resolvió el problema de desvanecimiento del gradiente en secuencias largas.',
    relatedMilestones: ["lstm-1997"],
    relatedTerms: ["rnn", "recurrent-network", "gating"],
  },

  "diffusion-model": {
    term: "Modelo de Difusión",
    definition:
      "Modelo generativo que aprende a eliminar ruido gradualmente de datos aleatorios. El entrenamiento enseña a predecir y remover ruido en múltiples pasos. En generación, comienza con ruido puro y lo refina iterativamente hasta producir imágenes de alta calidad. Usado en DALL-E 2, Stable Diffusion y Midjourney.",
    relatedMilestones: ["diffusion-2022"],
    relatedTerms: ["generative-ai", "denoising", "latent-space"],
  },

  "reinforcement-learning": {
    term: "Reinforcement Learning (Aprendizaje por Refuerzo)",
    definition:
      "Paradigma de ML donde un agente aprende a tomar decisiones mediante interacción con un entorno, recibiendo recompensas o castigos. El agente aprende una política que maximiza la recompensa acumulada a largo plazo. Usado en AlphaGo, robótica y videojuegos.",
    relatedMilestones: ["deepblue-1997"],
    relatedTerms: ["q-learning", "policy", "rlhf"],
  },
};
