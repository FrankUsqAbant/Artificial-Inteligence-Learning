/**
 * Rutas de Aprendizaje para la Línea de Tiempo de IA
 * Define caminos estructurados para diferentes niveles de conocimiento
 */

export interface LearningPath {
  id: string;
  title: string;
  description: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  milestones: string[]; // IDs de milestones en orden recomendado
  estimatedHours: number;
  prerequisites?: string[]; // IDs de otras rutas requeridas
  skills: string[]; // Habilidades que se adquirirán
  icon?: string; // Emoji opcional para visualización
}

export const LEARNING_PATHS: LearningPath[] = [
  {
    id: "foundations",
    title: "Fundamentos de IA: De Lógica a Máquinas Pensantes",
    description:
      "Comienza desde los fundamentos matemáticos y filosóficos de la IA hasta los primeros algoritmos de aprendizaje automático.",
    difficulty: "beginner",
    icon: "🎓",
    milestones: [
      "boole-1854", // Álgebra de Boole
      "turing-1950", // Test de Turing
      "dartmouth-1956", // Conferencia de Dartmouth
      "perceptron-1958", // Perceptrón
      "backprop-1986", // Backpropagation
    ],
    estimatedHours: 12,
    skills: [
      "Lógica booleana",
      "Conceptos básicos de IA",
      "Redes neuronales simples",
      "Algoritmos de aprendizaje",
    ],
  },

  {
    id: "deep-learning-revolution",
    title: "La Revolución del Deep Learning",
    description:
      "Explora cómo las redes neuronales profundas transformaron la IA moderna, desde LeNet hasta ResNet.",
    difficulty: "intermediate",
    icon: "🚀",
    milestones: [
      "lenet-1989", // LeNet: primeras CNNs
      "imagenet-2012", // AlexNet: inicio del deep learning moderno
      "gans-2014", // GANs: redes generativas adversariales
      "resnet-2015", // ResNet: redes residuales
    ],
    estimatedHours: 15,
    prerequisites: ["foundations"],
    skills: [
      "Redes convolucionales",
      "Transferlearning",
      "Arquitecturas profundas",
      "Generación de imágenes",
    ],
  },

  {
    id: "nlp-transformers",
    title: "NLP y la Era de los Transformers",
    description:
      "Domina el procesamiento de lenguaje natural desde Word2Vec hasta los modelos de lenguaje más avanzados.",
    difficulty: "intermediate",
    icon: "💬",
    milestones: [
      "word2vec-2013", // Embeddings de palabras
      "transformers-2017", // Arquitectura Transformer
      "chatgpt-2022", // ChatGPT y RLHF
    ],
    estimatedHours: 18,
    prerequisites: ["foundations"],
    skills: [
      "Embeddings",
      "Mecanismos de atención",
      "Transformers",
      "RLHF",
      "Prompting",
    ],
  },

  {
    id: "generative-ai",
    title: "IA Generativa: Creando Contenido con IA",
    description:
      "Aprende sobre modelos generativos desde GANs hasta modelos de difusión y IA multimodal.",
    difficulty: "advanced",
    icon: "🎨",
    milestones: [
      "gans-2014", // GANs
      "diffusion-2022", // Modelos de difusión
      "sora-2024", // Sora: video generativo
    ],
    estimatedHours: 20,
    prerequisites: ["deep-learning-revolution"],
    skills: [
      "GANs",
      "Difusión",
      "Generación de imagen/video",
      "Modelos multimodales",
    ],
  },

  {
    id: "ai-reasoning",
    title: "Fronteras de la IA: Reasoning y AGI",
    description:
      "Explora los avances más recientes en razonamiento, planificación y el camino hacia la inteligencia artificial general.",
    difficulty: "advanced",
    icon: "🧠",
    milestones: [
      "lora-2023", // LoRA: fine-tuning eficiente
      "reasoning-2025", // Modelos de razonamiento
    ],
    estimatedHours: 10,
    prerequisites: ["nlp-transformers", "generative-ai"],
    skills: ["Razonamiento", "Planificación", "AGI concepts", "Alignment"],
  },

  {
    id: "complete-timeline",
    title: "Cronología Completa: Toda la Historia de la IA",
    description:
      "Recorre toda la línea de tiempo de la IA desde 1837 hasta 2025, comprendiendo cada hito en su contexto histórico.",
    difficulty: "intermediate",
    icon: "📚",
    milestones: [
      "babbage-1837",
      "lovelace-1843",
      "boole-1854",
      "turing-1950",
      "dartmouth-1956",
      "perceptron-1958",
      "expert-1970",
      "backprop-1986",
      "lenet-1989",
      "deepblue-1997",
      "lstm-1997",
      "svm-1995",
      "imagenet-2012",
      "word2vec-2013",
      "gans-2014",
      "resnet-2015",
      "transformers-2017",
      "chatgpt-2022",
      "diffusion-2022",
      "lora-2023",
      "sora-2024",
      "reasoning-2025",
    ],
    estimatedHours: 40,
    skills: [
      "Historia completa de IA",
      "Evolución de paradigmas",
      "Contexto histórico",
      "Visión panorámica",
    ],
  },
];

/**
 * Obtiene una ruta de aprendizaje por ID
 */
export function getLearningPathById(id: string): LearningPath | undefined {
  return LEARNING_PATHS.find((path) => path.id === id);
}

/**
 * Obtiene rutas recomendadas para un nivel específico
 */
export function getLearningPathsByLevel(
  difficulty: "beginner" | "intermediate" | "advanced"
): LearningPath[] {
  return LEARNING_PATHS.filter((path) => path.difficulty === difficulty);
}

/**
 * Calcula el progreso de un usuario en una ruta
 */
export function calculatePathProgress(
  pathId: string,
  completedMilestones: string[]
): number {
  const path = getLearningPathById(pathId);
  if (!path) return 0;

  const completed = path.milestones.filter((id) =>
    completedMilestones.includes(id)
  ).length;
  return (completed / path.milestones.length) * 100;
}

/**
 * Obtiene el próximo milestone recomendado en una ruta
 */
export function getNextMilestoneInPath(
  pathId: string,
  completedMilestones: string[]
): string | null {
  const path = getLearningPathById(pathId);
  if (!path) return null;

  for (const milestoneId of path.milestones) {
    if (!completedMilestones.includes(milestoneId)) {
      return milestoneId;
    }
  }

  return null; // Ruta completada
}
