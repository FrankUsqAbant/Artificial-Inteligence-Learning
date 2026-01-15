# 🌌 Nexus AI Timeline

<div align="center">
  <img src="./assets/preview.png" alt="Nexus AI Timeline Preview" width="100%">
  
  <br>
  
  [![Live Demo](https://img.shields.io/badge/DEMO-VIVO-brightgreen?style=for-the-badge&logo=rocket&logoColor=white)](https://frankusqabant.github.io/Artificial-Inteligence-Learning/)
  
  **Explora la evolución de la Inteligencia Artificial desde la comodidad de tu navegador**
</div>

## 🎯 Sobre el Proyecto

**Nexus AI Timeline** es una plataforma educativa interactiva y autónoma que te lleva en un viaje fascinante a través de la historia completa de la Inteligencia Artificial, desde 1837 hasta 2025.

Este proyecto nació con una misión clara: **democratizar el conocimiento sobre IA** y proporcionar una experiencia de aprendizaje inmersiva, estructurada y 100% offline para cualquier persona interesada en entender cómo llegamos a la era de GPT-4, modelos de difusión y el camino hacia la AGI.

---

## ✨ ¿Qué hace este proyecto único?

### 📚 Plataforma Educativa Completa

- **22 hitos históricos** desde la Máquina Analítica (1837) hasta los Modelos de Razonamiento (2025)
- **15 milestones con deep-dive** (~800 palabras cada uno) que incluyen:
  - Contexto histórico profundo
  - Explicaciones técnicas detalladas
  - Código funcional (Python, PyTorch, NumPy)
  - Recursos para profundizar
- **~23,000 palabras** de contenido educativo original

### 🧠 Sistema de Conocimiento Local

- **100% autónomo**: Sin dependencias de APIs externas
- **40 FAQs** respondiendo preguntas comunes sobre IA
- **15 conceptos matemáticos** fundamentales (álgebra lineal, cálculo, probabilidad)
- **14 conceptos básicos** de IA explicados para principiantes
- **15 términos técnicos** en glosario

### 🎓 Rutas de Aprendizaje Estructuradas

- **6 rutas curadas** por nivel de dificultad:
  - 🎓 Fundamentos de IA (12h)
  - 🚀 Revolución Deep Learning (15h)
  - 💬 NLP y Transformers (18h)
  - 🎨 IA Generativa (20h)
  - 🧠 AI Reasoning y AGI (10h)
  - 📚 Cronología Completa (40h)

### 📊 Sistema de Progreso Gamificado

- Track de milestones completados con **localStorage**
- Sistema de **achievements** automático (🥉🥈🥇🏆)
- Milestones bloqueados/desbloqueados progresivamente
- Visualización de progreso por ruta

---

## 🚀 Características Técnicas

- ⚡ **React + TypeScript** - Código type-safe y modular
- 🎨 **Tailwind CSS** - Diseño moderno y responsive
- 🌐 **100% Client-side** - No requiere backend
- 📱 **Mobile-first** - Funciona en cualquier dispositivo
- 🔒 **Offline-first** - Toda la data es local
- ♿ **Accesible** - Pensado para todos

---

## 🎓 Objetivos Educativos

Este proyecto busca que cualquier persona, sin importar su nivel de conocimiento previo, pueda:

1. **Entender la historia completa de la IA**: Desde los fundamentos lógicos de Boole hasta los modelos de razonamiento de 2025
2. **Aprender los conceptos fundamentales**: Qué es ML, cómo funciona el entrenamiento, qué son las redes neuronales
3. **Dominar la terminología**: Transformers, GANs, embeddings, fine-tuning, RLHF, etc.
4. **Ver código funcional**: Implementaciones de Perceptrón, Backpropagation, CNNs, Transformers, GANs
5. **Seguir un camino estructurado**: Rutas de aprendizaje que guían desde principiante hasta avanzado
6. **Alcanzar ~88% de conocimiento en IA**: Cobertura comparable a un curso universitario introductorio-intermedio

---

## 💡 ¿Por qué construí esto?

La IA está transformando el mundo a una velocidad sin precedentes. Sin embargo, mucho del conocimiento está disperso en papers académicos, blogs técnicos, o encerrado en cursos costosos.

Quería crear una **única fuente de verdad** que:

- Sea **gratuita y accesible** para todos
- Funcione **offline** (sin depender de APIs o conexión estable)
- Presente la información de forma **narrativa e histórica** (no solo técnica)
- Incluya **código real** para aprender haciendo
- Sea **hermosa y motivadora** (diseño que inspire a seguir aprendiendo)

---

## 👨‍💻 Sobre el Autor

### Frank Abanto

Soy **desarrollador full-stack** y entusiasta de la Inteligencia Artificial, apasionado por la educación tecnológica y la creación de herramientas que democraticen el conocimiento.

Este proyecto es el resultado de meses de investigación, curación de contenido y desarrollo, con el objetivo de crear la mejor experiencia educativa sobre IA disponible de forma gratuita.

**Misión**: Hacer que el conocimiento sobre IA sea accesible, comprensible y motivador para cualquier persona que quiera aprender.

---

## 🌐 Conéctate Conmigo

¿Te gustó el proyecto? ¡Conectemos!

- 💼 **LinkedIn**: [Frank Abanto](https://linkedin.com/in/frankabanto)
- 📺 **YouTube**: [@frankabanto](https://youtube.com/@frankabanto)
- 📸 **Instagram**: [@frank_abant](https://instagram.com/frank_abant)
- 🐦 **Twitter/X**: [@FrankUsqAbanto](https://twitter.com/FrankUsqAbanto)

---

## 📊 Cobertura Educativa

| Área                    | Cobertura |
| ----------------------- | --------- |
| Historia y Contexto     | 95%       |
| Conceptos Fundamentales | 90%       |
| Terminología            | 92%       |
| Práctica con Código     | 75%       |
| Matemáticas             | 70%       |
| Aplicaciones            | 88%       |
| **TOTAL**               | **~88%**  |

---

## 🛠️ Instalación y Uso

```bash
# Clonar el repositorio
git clone https://github.com/frank-abanto/nexus-ai-timeline.git

# Navegar al directorio
cd nexus-ai-timeline

# Instalar dependencias
npm install
# o
yarn install

# Iniciar el servidor de desarrollo
npm run dev
# o
yarn dev

# Abrir http://localhost:5173
```

---

## 📚 Estructura del Proyecto

```
nexus-ai-timeline/
├── components/          # Componentes React
│   ├── Dashboard.tsx
│   ├── LessonDrawer.tsx
│   ├── NexusOracle.tsx
│   └── LearningPathVisualizer.tsx
├── data/               # Base de datos local
│   ├── ai_timeline.ts       # 22 milestones históricos
│   ├── faq_database.ts      # 40 FAQs
│   ├── glossary.ts          # 15 términos técnicos
│   ├── fundamental_concepts.ts  # 14 conceptos básicos
│   ├── mathematical_foundations.ts  # 15 conceptos matemáticos
│   └── learning_paths.ts    # 6 rutas de aprendizaje
├── services/           # Lógica de negocio
│   └── localKnowledgeService.ts  # Motor de búsqueda local
├── hooks/              # Custom hooks
│   └── useUserProgress.ts   # Tracking de progreso
└── assets/             # Imágenes y recursos
```

---

## 🎯 Roadmap Futuro

- [ ] **Modo oscuro/claro** dinámico
- [ ] **Quizzes interactivos** (220 preguntas)
- [ ] **Exportar progreso** a PDF
- [ ] **Búsqueda con sugerencias** en tiempo real
- [ ] **Más milestones expandidos** (objetivo: 22/22)
- [ ] **Guías de implementación** paso a paso
- [ ] **Modo de lectura** sin distracciones
- [ ] **Síntesis de voz** mejorada

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la [Licencia MIT](LICENSE).

---

## 🙏 Agradecimientos

A todos los pioneros de la IA que hicieron posible este campo fascinante:

- Alan Turing, John McCarthy, Marvin Minsky
- Geoffrey Hinton, Yann LeCun, Yoshua Bengio
- Ian Goodfellow, Ilya Sutskever, Andrej Karpathy
- Y a toda la comunidad open source

---

## 💖 Apoya el Proyecto

Si este proyecto te resultó útil, considera:

- ⭐ Darle una estrella en GitHub
- 🐛 Reportar bugs o sugerir mejoras
- 📢 Compartirlo con amigos interesados en IA
- 💬 Dejar feedback en las issues

---

<div align="center">

**Hecho con ❤️ por Frank Abanto**

_Aprende IA. Entiende el pasado. Construye el futuro._

</div>
