import { Milestone } from "../types";

export const AI_TIMELINE: Milestone[] = [
  {
    id: "babbage-1837",
    year: 1837,
    era: "Pre-Computing",
    type: "milestone",
    title: "Máquina Analítica",
    description:
      "Charles Babbage diseña el primer computador mecánico de propósito general.",
    historical_context:
      "Babbage propuso una unidad lógica, memoria y control por tarjetas perforadas.",
    legacy: "El diseño teórico fundamental de toda computadora moderna.",
    theory: {
      concept: "Arquitectura de Computación.",
      math: "Lógica Programable",
      analogy: "Un relojero intentando construir un cerebro de engranajes.",
    },
    practice: {
      challenge:
        "¿Cómo se llamaba el dispositivo de memoria en la máquina de Babbage?",
      starter_code: "",
      expected_logic: "almacén",
    },
  },
  {
    id: "lovelace-1843",
    year: 1843,
    era: "Pre-Computing",
    type: "milestone",
    title: "El Primer Algoritmo",
    description:
      "Ada Lovelace visualiza máquinas que pueden procesar más que solo números.",
    historical_context:
      "Ada se dio cuenta de que si una máquina podía manipular símbolos, podía crear música o arte.",
    legacy:
      "Se le considera la visionaria original de la computación universal.",
    theory: {
      concept: "Computación Simbólica.",
      math: "B_n = - \frac{1}{n+1} sum_{k=0}^{n-1} \binom{n+1}{k} B_k",
      analogy: "Un telar que teje patrones de información.",
    },
    practice: {
      challenge: "¿Cómo llamó Ada Lovelace a su enfoque?",
      starter_code: "",
      expected_logic: "ciencia poética",
    },
  },
  {
    id: "boole-1854",
    year: 1854,
    era: "Pre-Computing",
    type: "milestone",
    title: "Álgebra de Boole",
    description:
      "George Boole publica las leyes del pensamiento, reduciendo la lógica a matemáticas.",
    historical_context:
      "Estableció que los razonamientos pueden expresarse mediante variables binarias.",
    legacy: "La base absoluta de la computación digital y las puertas lógicas.",
    theory: {
      concept: "Lógica Binaria.",
      math: "A land B, A lor B, \neg A",
      analogy: "Un sistema de tuberías donde el agua fluye (1) o no fluye (0).",
    },
    practice: {
      challenge: "¿Cuáles son los dos únicos valores en el álgebra de Boole?",
      starter_code: "",
      expected_logic: "verdadero y falso",
    },
  },
  {
    id: "turing-1950",
    year: 1950,
    era: "Symbolic Era",
    type: "era-change",
    title: "El Test de Turing",
    description: 'Alan Turing propone: "¿Pueden pensar las máquinas?"',
    historical_context:
      "Buscaba una medida objetiva de inteligencia basada en el comportamiento.",
    legacy: "Estándar de oro para medir la capacidad conversacional.",
    theory: {
      concept: "Funcionalismo Computacional.",
      math: "P(M|T) > 0.5",
      analogy: "Un actor que engaña a su propia familia bajo un disfraz.",
    },
    practice: {
      challenge: "¿Nombre original del experimento de Turing?",
      starter_code: "",
      expected_logic: "juego de la imitación",
    },
  },
  {
    id: "dartmouth-1956",
    year: 1956,
    era: "Symbolic Era",
    type: "milestone",
    title: "Conferencia de Dartmouth",
    description: 'Se acuña el término oficial "Inteligencia Artificial".',
    historical_context:
      "McCarthy y Minsky creían que la IA se resolvería en un verano.",
    legacy: "Estableció la IA como campo científico independiente.",
    theory: {
      concept: "IA Simbólica (GOFAI).",
      math: "If (A ∧ B) → C",
      analogy: "Un bibliotecario siguiendo un manual perfecto.",
    },
    practice: {
      challenge: "¿Quién acuñó el término AI en 1956?",
      starter_code: "",
      expected_logic: "John McCarthy",
    },
    resources: ["https://en.wikipedia.org/wiki/Dartmouth_workshop"],
    deep_dive: `
## El Nacimiento Oficial de la Inteligencia Artificial

El verano de 1956 marcó un momento decisivo en la historia de la ciencia. Del **1 de julio al 31 de agosto**, en el tranquilo campus del Dartmouth College en Hanover, New Hampshire, se reunió un grupo extraordinario de científicos para un **taller de investigación de 8 semanas** que cambiaría el mundo para siempre.

Este evento, conocido como la **Conferencia de Dartmouth**, es considerado el nacimiento oficial del campo de la Inteligencia Artificial.

### Los Padres Fundadores

La conferencia fue organizada por **John McCarthy**, un joven matemático de Stanford, quien acuñó el término "Inteligencia Artificial" en la propuesta original. McCarthy reunió a un grupo estelar de 10 participantes:

**Los Organizadores Principales**:
1. **John McCarthy** (Dartmouth) - Creador del término "IA" y del lenguaje LISP
2. **Marvin Minsky** (Harvard/MIT) - Cofundador del MIT AI Lab
3. **Nathaniel Rochester** (IBM) - Diseñador de la IBM 701
4. **Claude Shannon** (Bell Labs) - Padre de la teoría de la información

**Participantes Clave**:
5. **Allen Newell** (RAND Corporation) - Co-creador de Logic Theorist
6. **Herbert Simon** (Carnegie Tech) - Premio Nobel, pionero en IA
7. **Arthur Samuel** (IBM) - Pionero en aprendizaje automático
8. **Ray Solomonoff** - Teoría algorítmica de la información
9. **Oliver Selfridge** (MIT) - Sistemas de reconocimiento de patrones
10. **Trenchard More** (Princeton) - Lógica matemática

### La Propuesta que lo Inició Todo

En 1955, McCarthy, Minsky, Rochester y Shannon redactaron una propuesta audaz pidiendo $13,500 a la Fundación Rockefeller:

> **"Proponemos que se realice un estudio de 2 meses y 10 personas sobre inteligencia artificial durante el verano de 1956 en Dartmouth College [...] El estudio se basa en la conjetura de que cada aspecto del aprendizaje o cualquier otra característica de la inteligencia puede, en principio, describirse con tanta precisión que una máquina puede ser programada para simularlo."**

Esta frase contenía una promesa revolucionaria: **toda la inteligencia humana podría ser formalizada y replicada en máquinas**.

### El Optimismo Inicial

La conferencia estuvo marcada por un optimismo extraordinario. Los investigadores creían que estaban a pocos años—quizás décadas como máximo—de crear máquinas verdaderamente inteligentes.

**Logros presentados en Dartmouth**:

1. **Logic Theorist** (Newell & Simon): El primer programa de IA que demostró teoremas matemáticos. Probó 38 de los primeros 52 teoremas del *Principia Mathematica* de Russell y Whitehead. ¡Incluso encontró una prueba más elegante que la original para el teorema 2.85!

2. **Programas de ajedrez**: Samuel presentó su programa de damas que aprendía jugando contra sí mismo.

3. **Procesamiento de lenguaje natural**: Primeras ideas sobre cómo hacer que las máquinas "entendieran" el lenguaje humano.

**Predicciones optimistas**:
- McCarthy y Minsky predijeron que en **20 años** (para 1976), las máquinas alcanzarían la inteligencia humana general.
- Herbert Simon declaró en 1965: "Las máquinas serán capaces, dentro de veinte años, de hacer cualquier trabajo que un hombre pueda hacer."

Este optimismo atrajo **financiamiento masivo** de DARPA (Agencia de Proyectos de Investigación Avanzada de Defensa) y otras agencias gubernamentales durante los años 60.

### El Primer Invierno de IA (1974-1980)

Pero la realidad fue menos generosa. Para principios de los 70, quedó claro que las promesas no se cumplirían tan rápido:

**Problemas fundamentales**:
- **Complejidad combinatoria**: Muchos problemas eran computacionalmente intratables.
- **Limitaciones de hardware**: Las computadoras eran demasiado lentas y con poca memoria.
- **Conocimiento del mundo**: Las máquinas carecían del "sentido común" humano.

**El Informe Lighthill (1973)**: El gobierno británico encargó al matemático James Lighthill una revisión de la IA. Su informe fue devastador: concluyó que la IA había fracasado en cumplir sus "grandiosas predicciones" y recomendó recortar drásticamente el financiamiento.

**Consecuencias**:
- 1974-1980: El **Primer Invierno de IA** - Financiamiento casi desapareció.
- Proyectos cancelados, laboratorios cerrados.
- "IA" se convirtió en una palabra tóxica en propuestas de investigación.

### El Legado Duradero

A pesar del fracaso de las predicciones iniciales, Dartmouth estableció los fundamentos del campo:

**Contribuciones conceptuales**:
- Definió la IA como un campo de estudio legítimo
- Estableció objetivos claros: razonamiento, aprendizaje, lenguaje, percepción
- Creó una comunidad de investigadores

**Tecnologías semilla**:
- LISP (McCarthy, 1958) - El lenguaje de programación de IA por décadas
- Heurística y búsqueda - Algoritmos fundamentales
- Representación del conocimiento - Cómo codificar información en máquinas

**Lecciones aprendidas**:
- La IA es más difícil de lo que parece
- El "sentido común" humano es extraordinariamente complejo
- El progreso requiere décadas, no años

### Conexión con el Presente (2025)

Casi **70 años después**, finalmente estamos viviendo versiones de lo que los fundadores imaginaron:

- **GPT-4 y o1**: Modelos de lenguaje que pasan versiones del Test de Turing
- **AlphaGo/AlphaZero**: Dominan juegos de estrategia
- **Vehículos autónomos**: Percepción y toma de decisiones
- **AGI en el horizonte**: Algunos expertos predicen AGI para 2030-2040

La visión de Dartmouth era correcta; solo la **línea de tiempo** estaba equivocada. Lo que ellos pensaron tomaría 20 años, ha tomado 70... y aún no hemos terminado.

## Timeline Post-Dartmouth

- **1956-1974**: Era Dorada - Optimismo y financiamiento abundante
- **1974-1980**: Primer Invierno de IA - Recortes y desilusión
- **1980-1987**: Auge de Sistemas Expertos - Resurgimiento comercial
- **1987-1993**: Segundo Invierno de IA - Colapso de sistemas expertos
- **1997+**: Era Moderna - Deep Learning y renacimiento

## Recursos para Profundizar

- **Propuesta original**: McCarthy et al. (1955). "A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence"
- **Libro**: McCorduck, P. "Machines Who Think" (1979) - Historia de la IA
- **Documental**: "Do You Trust This Computer?" (2018) - Incluye historia de Dartmouth
- **Artículo**: "The Dartmouth College Artificial Intelligence Conference: The Next Fifty Years" (2006)
    `,
  },
  {
    id: "perceptron-1958",
    year: 1958,
    era: "Connectionist Era",
    type: "milestone",
    title: "El Perceptrón",
    description: "Frank Rosenblatt crea la primera red neuronal que aprende.",
    historical_context: "Inspirado en la biología neuronal.",
    legacy: "Abuelo de todo el Deep Learning moderno.",
    theory: {
      concept: "Ajuste de Pesos.",
      math: "f(x) = \\text{sign}(\\mathbf{w} \\cdot \\mathbf{x} + b)",
      analogy: "Un interruptor que aprende cuánta presión necesitas.",
    },
    practice: {
      challenge: "¿Qué se ajusta en una neurona para aprender?",
      starter_code: "",
      expected_logic: "pesos",
    },
    deep_dive: `## El Nacimiento de las Redes Neuronales

En 1958, el psicólogo Frank Rosenblatt creó el **Perceptrón**, la primera máquina capaz de aprender por sí misma. Inspirado por las investigaciones de Warren McCulloch y Walter Pitts sobre neuronas artificiales (1943), Rosenblatt quería construir una máquina que imitara la forma en que el cerebro humano aprende.

### ¿Qué es un Perceptrón?

El perceptrón es el modelo más simple de una neurona artificial. Funciona así:

1. **Recibe múltiples entradas** (como píxeles de una imagen)
2. **Multiplica cada entrada por un peso** (importancia de esa entrada)
3. **Suma todos los valores ponderados**
4. **Aplica una función de decisión** (activación): si la suma supera un umbral, "dispara" (salida = 1), sino no (salida = 0)

### La Fórmula Matemática

\`\`\`
y = f(w₁x₁ + w₂x₂ + ...  + wₙxₙ + b)

donde:
- x₁, x₂, ..., xₙ = entradas
- w₁, w₂, ..., wₙ = pesos (qué tan importante es cada entrada)
- b = bias (sesgo, umbral de activación)
- f = función escalón (sign function)
\`\`\`

### Ejemplo en Python

\`\`\`python
import numpy as np

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
        self.learning_rate = 0.1
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0
    
    def train(self, training_data, labels, epochs):
        for epoch in range(epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Clasificador de manzanas vs naranjas
fruits = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])  # [color, textura]
labels = np.array([0, 0, 1, 1])  # 0=manzana, 1=naranja

perceptron = Perceptron(num_inputs=2)
perceptron.train(fruits, labels, epochs=10)
print(f"Predicción: {perceptron.predict([1, 1])}")  # → 1 (naranja)
\`\`\`

### Limitaciones

En 1969, Minsky y Papert demostraron que el perceptrón **solo resuelve problemas linealmente separables**. No puede aprender el operador XOR, lo que causó el primer "Invierno de la IA".

### Legado

El perceptrón estableció los principios que perduran hoy: aprendizaje supervisado, ajuste de pesos mediante gradiente descendente, y la idea de que redes multicapa pueden superar sus limitaciones. Todas las redes neuronales modernas (CNNs, Transformers, GPT) son descendientes directos del perceptrón.

### Recursos

- **Paper original**: "The Perceptron: A Probabilistic Model" (Rosenblatt, 1958)
- **Video**: "But what IS a neural network?" - 3Blue1Brown (YouTube)
- **Interactivo**: TensorFlow Playground (playground.tensorflow.org)
`,
  },
  {
    id: "expert-1970",
    year: 1970,
    era: "Symbolic Era",
    type: "milestone",
    title: "Sistemas Expertos",
    description:
      "IAs basadas en reglas que emulan el juicio de un especialista.",
    historical_context:
      "MYCIN podía diagnosticar infecciones de sangre mejor que algunos médicos.",
    legacy: "Primer éxito comercial masivo de la IA.",
    theory: {
      concept: "Motor de Inferencia.",
      math: "Ruleset: {If A dots then B}",
      analogy: "Un árbol de decisión gigante que nunca olvida una regla.",
    },
    practice: {
      challenge: "¿Cómo se llama la parte que procesa las reglas?",
      starter_code: "",
      expected_logic: "motor de inferencia",
    },
  },
  {
    id: "backprop-1986",
    year: 1986,
    era: "Connectionist Era",
    type: "milestone",
    title: "Retropropagación",
    description: "Algoritmo que permite entrenar redes con muchas capas.",
    historical_context:
      "Hinton demuestra que las redes pueden aprender conceptos complejos.",
    legacy: "Fin del invierno de la IA.",
    theory: {
      concept: "Gradiente Descendente.",
      math: "\\frac{\\partial E}{\\partial w}",
      analogy: "Un profesor dando feedback específico a cada alumno.",
    },
    practice: {
      challenge: "¿Cómo se llama el proceso de enviar el error hacia atrás?",
      starter_code: "",
      expected_logic: "backpropagation",
    },
    deep_dive: `## La Revolución que Resucitó la IA

En 1986, David Rumelhart, Geoffrey Hinton y Ronald Williams publicaron un paper que cambiaría todo: "Learning representations by back-propagating errors". Este algoritmo permitió entrenar redes neuronales con **múltiples capas ocultas**, resolviendo las limitaciones del perceptrón simple.

### El Problema que Resolvió

Antes de backpropagation, nadie sabía cómo **ajustar los pesos de las capas intermedias** en una red profunda. Solo se podía entrenar el perceptrón de una capa. ¿Cómo saber qué tanto contribuye cada neurona oculta al error final?

### ¿Cómo Funciona?

Backpropagation usa la **regla de la cadena del cálculo** para calcular el gradiente del error respecto a cada peso de la red, propagando el error desde la salida hacia atrás:

**Fase Forward** (hacia adelante):
1. Las entradas se propagan capa por capa
2. Cada neurona calcula: \`activación = f(Σ(peso × entrada) + bias)\`
3. Se obtiene la predicción final

**Fase Backward** (hacia atrás):
1. Se calcula el error: \`error = predicción - valor_real\`
2. El error se propaga hacia atrás usando la regla de la cadena
3. Se calculan los gradientes: \`∂E/∂w\` para cada peso
4. Se actualizan los pesos: \`w_nuevo = w_viejo - learning_rate × gradiente\`

### Matemáticas Simplificadas

Para una neurona en la capa L:

\`\`\`
δᴸ = (aᴸ - y) ⊙ σ'(zᴸ)     ← Error en capa de salida
δˡ = ((wˡ⁺¹)ᵀ δˡ⁺¹) ⊙ σ'(zˡ)  ← Error propagado a capa anterior

∂E/∂w = aˡ⁻¹ × δˡ           ← Gradiente del peso
w ← w - η × ∂E/∂w             ← Actualización de peso
\`\`\`

Donde:
- δ = delta (error local de la neurona)
- σ' = derivada de la función de activación
- ⊙ = producto elemento a elemento (Hadamard)
- η = learning rate (tasa de aprendizaje)

### Implementaci ón en Python con NumPy

\`\`\`python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar pesos aleatorios
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        # Propagación hacia adelante
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate=0.1):
        # BACKPROPAGATION
        m = X.shape[0]
        
        # Error en capa de salida
        delta2 = (self.a2 - y) * sigmoid_derivative(self.a2)
        
        # Gradientes de capa de salida
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        
        # Propagar error hacia atrás
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)
        
        # Gradientes de capa oculta
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m
        
        # Actualizar pesos
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            if epoch % 1000 == 0:
                loss = np.mean((y - output) ** 2)
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Ejemplo: Aprender XOR (imposible para perceptrón simple!)
X = np.array([[0,0], [0,1], [1,0], [1,1 ]])
y = np.array([[0], [1], [1], [0]])  # XOR

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=10000)

# Probar predicciones
predictions = nn.forward(X)
print("\\nPredicciones finales:")
for i, pred in enumerate(predictions):
    print(f"{X[i]} → {pred[0]:.3f} (esperado: {y[i][0]})")
\`\`\`

### El Impacto Histórico

Backpropagation **terminó el Invierno de la IA** de los años 70-80. Demostró que las redes neuronales **sí podían aprender funciones complejas y no lineales** como el XOR, que era imposible para el perceptrón simple.

### Por Qué Importa Hoy

**Todas las redes neuronales modernas usan backpropagation**:
- CNNs para visión por computadora
- RNNs y Transformers para lenguaje
- GANs para generación de imágenes
- Redes profundas con millones de parámetros

Sin backpropagation, no existirían: ImageNet, GPT, DALL-E, AlphaGo, ChatGPT, ni ningún sistema de deep learning moderno.

### Recursos para Profundizar

- **Paper original**: "Learning representations by back-propagating errors" (Rumelhart, Hinton, Williams, 1986)
- **Video visual**: "Backpropagation calculus" - 3Blue1Brown (YouTube)
- **Curso interactivo**: "Neural Networks and Deep Learning" - Andrew Ng (Coursera)
- **Libro**: "Deep Learning" - Goodfellow, Bengio, Courville (Capítulo 6)
`,
  },
  {
    id: "lenet-1989",
    year: 1989,
    era: "Connectionist Era",
    type: "milestone",
    title: "LeNet-5",
    description:
      "Yann LeCun aplica convoluciones para leer números escritos a mano.",
    historical_context:
      "Fue la primera aplicación comercial exitosa de redes neuronales en bancos.",
    legacy: "Sentó las bases de la visión por computadora moderna.",
    theory: {
      concept: "Convoluciones (CNN).",
      math: "S(i,j) = (I * K)(i,j)",
      analogy:
        "Una linterna que recorre una imagen buscando patrones geométricos.",
    },
    practice: {
      challenge: "¿Qué tipo de capas inventó LeCun para imágenes?",
      starter_code: "",
      expected_logic: "convoluciones",
    },
    resources: ["http://yann.lecun.com/exdb/lenet/"],
    deep_dive: `
## La Primera CNN que Funcionó en el Mundo Real

En 1989, Yann LeCun, trabajando en los laboratorios Bell de AT&T, creó **LeNet**, la primera Red Neuronal Convolucional (CNN) que demostró ser práctica para aplicaciones del mundo real. Su objetivo era aparentemente simple: **leer dígitos escritos a mano en cheques bancarios**.

Pero LeNet no solo resolvió ese problema—estableció el blueprint arquitectónico que 23 años después revolucionaría la IA con AlexNet (2012).

### El Problema: Reconocimiento de Dígitos

En los años 80-90, los bancos procesaban **millones de cheques diariamente**. Cada cheque tenía que ser leído manualmente para extraer la cantidad escrita—un proceso lento, costoso y propenso a errores.

**El desafío técnico**:
- Cada persona escribe números de forma diferente
- La escritura puede estar inclinada, borrosa o parcialmente oculta
- Necesitas 99.9%+ de precisión (errores cuestan dinero)

LeCun se inspiró en el trabajo de **Kunihiko Fukushima** sobre el **Neocognitron** (1980), una red neuronal que imitaba la corteza visual del cerebro. Fukushima había descubierto que las neuronas visuales tienen "campos receptivos locales"—responden solo a pequeñas regiones de la imagen.

### La Arquitectura LeNet-5

**LeNet-5** (versión de 1998, refinamiento de la original de 1989) tiene una arquitectura elegantemente simple:

\`\`\`
INPUT (32x32 imagen) 
  ↓
CONV1 (6 filtros 5x5) → ReLU → POOL (2x2)
  ↓
CONV2 (16 filtros 5x5) → ReLU → POOL (2x2)
  ↓
FLATTEN
  ↓
FC1 (120 neuronas) → ReLU
  ↓
FC2 (84 neuronas) → ReLU
  ↓
OUTPUT (10 clases: dígitos 0-9)
\`\`\`

**Innovaciones clave**:

1. **Capas Convolucionales**: En lugar de conectar cada píxel a cada neurona (fully connected), LeNet usa "filtros" que se deslizan sobre la imagen detectando características locales como bordes, curvas.

2. **Compartir parámetros**: El mismo filtro se aplica a toda la imagen. Esto reduce drásticamente el número de parámetros a aprender.

3. **Pooling (Subsampling)**: Reduce el tamaño de la imagen gradualmente, manteniendo las características importantes pero reduciendo cómputo.

4. **Jerarquía de características**:
   - **Capa 1**: Detecta bordes simples (horizontal, vertical, diagonal)
   - **Capa 2**: Combina bordes en formas (curvas, esquinas)
   - **Capas finales**: Reconocen dígitos completos

### El Dataset: MNIST

LeCun y su equipo crearon el **MNIST** (Modified National Institute of Standards and Technology), el dataset de machine learning MÁS famoso de la historia:

- **60,000 imágenes de entrenamiento**
- **10,000 imágenes de prueba**
- Dígitos escritos a mano (0-9)
- Imágenes de 28x28 píxeles en escala de grises

MNIST se convirtió en el "Hello World" del deep learning. Si querías probar una nueva arquitectura de red neuronal, primero la probabas en MNIST.

**Resultado de LeNet-5 en MNIST**: ~99.2% de precisión—revolucionario para 1998.

### Aplicación Comercial: Leyendo Cheques

LeNet fue **implementado comercialmente** por bancos americanos en los 90-2000s:

- Procesaba **millones de cheques al día**
- Reducción de costos operativos significativa
- Una de las primeras aplicaciones exitosas de deep learning

**Por qué fue posible solo entonces**:
- **Hardware**: Las computadoras finalmente tenían suficiente RAM y CPU
- **Datos**: Millones de imágenes de cheques reales para entrenar
- **Algoritmo**: Backpropagation (1986) hizo posible entrenar redes profundas

### Diferencias con CNNs Modernas

LeNet era simple comparado con arquitecturas actuales:

| Aspecto | LeNet-5 (1998) | AlexNet (2012) | ResNet-50 (2015) |
|---------|----------------|----------------|------------------|
| Capas | 7 | 8 | 50 |
| Parámetros | ~60,000 | ~60 millones | ~25 millones |
| Activación | Tanh | ReLU | ReLU |
| Regularización | No | Dropout | Batch Norm |
| Hardware | CPU | 2 GPUs | Múltiples GPUs |

**Lo que faltaba en LeNet**:
- **ReLU**: LeNet usó tanh/sigmoid (más lentas de entrenar)
- **Dropout**: No había regularización explícita → overfitting
- **Batch Normalization**: No se conocía aún
- **GPUs**: Entrenamiento era extremadamente lento

### El "Segundo Invierno" y el Olvido

A pesar del éxito de LeNet, las redes neuronales cayeron nuevamente en desgracia en los 2000s:

**Razones**:
- **SVMs (Support Vector Machines)** eran más fáciles de entrenar y daban buenos resultados
- **Random Forests** y otros métodos ensemble dominaron competencias
- Las redes neuronales eran vistas como "demasiado complicadas" y "caja negra"

Yann LeCun siguió creyendo en las CNNs incluso cuando nadie más lo hacía. Su perseverancia fue vindicada en **2012 con AlexNet**, que demostró que LeNet tenía razón todo el tiempo—solo necesitaba más datos, más cómputo (GPUs), y algunas mejoras (ReLU, Dropout).

### Legado e Impacto

LeNet estableció patrones arquitectónicos que persisten hoy:

✅ **Convolución → Activación → Pooling** (el patrón básico)
✅ **Jerarquía de características** (de simple a complejo)
✅ **Reducción progresiva de dimensión espacial**
✅ **Compartir parámetros** (weight sharing)

**Aplicaciones herederas**:
- **Visión por computadora**: Detección de objetos, segmentación, reconocimiento facial
- **OCR moderno**: Google Lens, escaneo de documentos
- **Vehículos autónomos**: Detección de señales, peatones
- **Medicina**: Análisis de rayos X, detección de tumores

Sin LeNet, no habría AlexNet. Sin AlexNet, no habría la revolución del deep learning.

## Implementación en PyTorch

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn

.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Crear y probar modelo
model = LeNet5()
dummy_input = torch.randn(1, 1, 28, 28)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # (1, 10)
\`\`\`

## Recursos para Profundizar

- **Paper original**: LeCun, Y. et al. (1998). "Gradient-Based Learning Applied to Document Recognition"
- **Dataset MNIST**: [yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- **Tutorial PyTorch**: Implementación moderna de LeNet-5
    `,
  },
  {
    id: "svm-1995",
    year: 1995,
    era: "Statistical Era",
    type: "milestone",
    title: "Máquinas de Soporte Vectorial",
    description:
      "Vapnik populariza el clasificador más robusto antes del Deep Learning.",
    historical_context:
      "Las SVM dominaron el ML por su base matemática sólida.",
    legacy: "Estándar para clasificación de alta precisión.",
    theory: {
      concept: "Margen Máximo.",
      math: "max \frac{2}{|mathbf{w}|}$",
      analogy:
        "Dibujar la frontera más ancha posible entre dos grupos de gente.",
    },
    practice: {
      challenge:
        "¿Cómo se llama la técnica para mover datos a más dimensiones en SVM?",
      starter_code: "",
      expected_logic: "kernel trick",
    },
  },
  {
    id: "lstm-1997",
    year: 1997,
    era: "Statistical Era",
    type: "milestone",
    title: "Memoria LSTM",
    description: "Hochreiter y Schmidhuber resuelven el olvido en redes.",
    historical_context: "Las redes antes no podían recordar secuencias largas.",
    legacy: "Permitió entender videos y voz.",
    theory: {
      concept: "Gating Mechanisms.",
      math: "f_t = sigma(W_f cdot [h_{t-1}, x_t] + b_f)",
      analogy: "Un post-it que decide qué información borrar de la memoria.",
    },
    practice: {
      challenge: "¿Qué significa la F en las puertas de una LSTM?",
      starter_code: "",
      expected_logic: "forget gate",
    },
  },
  {
    id: "deepblue-1997",
    year: 1997,
    era: "Statistical Era",
    type: "milestone",
    title: "Deep Blue vs Kasparov",
    description: "IBM derrota al campeón mundial de ajedrez.",
    historical_context: "Hito cultural de la potencia de cálculo bruta.",
    legacy: "Demostró que algoritmos de búsqueda superan la intuición.",
    theory: {
      concept: "Búsqueda Heurística.",
      math: "Alpha-Beta Pruning",
      analogy: "Un calculista que ve 200 millones de futuros por segundo.",
    },
    practice: {
      challenge: "¿A qué campeón derrotó Deep Blue?",
      starter_code: "",
      expected_logic: "Kasparov",
    },
  },
  {
    id: "imagenet-2012",
    year: 2012,
    era: "Deep Learning Era",
    type: "era-change",
    title: "Revolución AlexNet",
    description: "El Deep Learning destroza los récords de visión.",
    historical_context: "Uso masivo de GPUs para entrenamiento paralelo.",
    legacy: "Inicio de la explosión moderna del Deep Learning.",
    theory: {
      concept: "Arquitectura Deep CNN.",
      math: "Softmax Classifier",
      analogy: "Un cerebro artificial con esteroides de tarjetas de video.",
    },
    practice: {
      challenge: "¿Qué hardware aceleró esta revolución?",
      starter_code: "",
      expected_logic: "GPU",
    },
    deep_dive: `## El Momento que Cambió la IA Moderna

El 30 de septiembre de 2012, un equipo de la Universidad de Toronto liderado por Alex Krizhevsky, Ilya Sutskever y Geoffrey Hinton participó en el desafío **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**. Su red neuronal, **AlexNet**, **aplastó** la competencia con un error del 15.3%, comparado con el 26.2% del segundo lugar. Esta victoria marcó el inicio de la era moderna del deep learning.

### El Desafío ImageNet

**ImageNet** es un dataset masivo con:
- **14+ millones** de imágenes etiquetadas a mano
- **20,000+ categorías** (perros, gatos, aviones, etc.)
- El desafío ILSVRC usa **1,000 categorías** y **1.2 millones de imágenes para entrenamiento**

Antes de 2012, los mejores sistemas usaban **características hechas a mano** (SIFT, HOG) + clasificadores tradicionales (SVM). Mejoraban ~1% por año.

AlexNet mejoró **10.8%** de un año a otro. Fue un salto nunca antes visto.

### La Arquitectura de AlexNet

AlexNet fue la primera CNN profunda exitosa en ImageNet:

\`\`\`
Input: Imagen 224×224×3 (RGB)
   ↓
Conv1: 96 filtros 11×11, stride 4 → ReLU → MaxPool
   ↓
Conv2: 256 filtros 5×5 → ReLU → MaxPool
   ↓
Conv3: 384 filtros 3×3 → ReLU
   ↓
Conv4: 384 filtros 3×3 → ReLU
   ↓
Conv5: 256 filtros 3×3 → ReLU → MaxPool
   ↓
FC6: 4096 neuronas → ReLU → Dropout
   ↓
FC7: 4096 neuronas → ReLU → Dropout
   ↓
FC8: 1000 neuronas (softmax) → Clasificación
\`\`\`

**Innovaciones clave**:
1. **ReLU**: \`f(x) = max(0, x)\` en lugar de sigmoid/tanh (entrena 6x más rápido)
2. **Dropout**: Apaga neuronas aleatoriamente durante entrenamiento para evitar overfitting
3. **Data Augmentation**: Voltear, recortar, cambiar colores de imágenes
4. **GPUs**: Entrenó en **2 GPUs NVIDIA GTX 580** durante **5-6 días**

### Código Conceptual en PyTorch

\`\`\`python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten para FC layers
        x = self.classifier(x)
        return x

# Crear modelo
model = AlexNet(num_classes=1000)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
# → ~60 millones de parámetros

# Ejemplo de uso
input_image = torch.randn(1, 3, 224, 224)  # Batch de 1 imagen
output = model(input_image)
print(output.shape)  # → torch.Size([1, 1000])

# Obtener predicción
probabilities = torch.nn.functional.softmax(output,  dim=1)
top_class = torch.argmax(probabilities)
print(f"Clase predicha: {top_class.item()}")
\`\`\`

### ¿Por Qué Funcionaron las GPUs?

**CPUs**: Procesadores generales, pocas cores (~8), secuenciales  
**GPUs**: Miles de cores simples, diseñadas para operaciones paralelas

Las convoluciones son **altamente paralelizables** (cada filtro opera independientemente). Las GPUs aceleran el entrenamiento **10-50x**.

\`\`\`
Matriz de imagen × Filtros convolucionales
     ↓
Miles de multiplicaciones simultáneas en GPU
     ↓
Backpropagation también paralelo
\`\`\`

Sin GPUs, entrenar AlexNet habría tomado **meses** en lugar de días.

### El Impacto Sísmico

AlexNet desencadenó una revolución:

**Investigación**:
- Todos empezaron a usar deep learning para visión
- Aparecieron arquitecturas más profundas: VGG (19 capas), ResNet (152 capas)
- Transfer learning: Usar redes pre-entrenadas en ImageNet

**Industria**:
- Facebook usa CNNs para reconocimiento facial
- Google Photos organiza millones de fotos automáticamente
- Autos autónomos usan CNNs para "ver" el mundo
- Diagnóstico médico con rayos X y resonancias

**Hardware**:
- NVIDIA se convirtió en líder de IA (sus acciones subieron 100x)
- Aparecieron chips especializados: TPUs de Google, NPUs

### Cronología Post-AlexNet

- **2014**: VGG + GoogleNet (Inception)
- **2015**: ResNet (redes residuales, 152 capas)
- **2017**: Vision Transformers empiezan a competir con CNNs
- **2024**: CNNs siguen siendo fundamentales en aplicaciones productivas

### Recursos para Profundizar

- **Paper original**: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012)
- **Dataset**: ImageNet (image-net.org)
- **Tutorial**: "CNN Explainer" - Herramienta interactiva (poloclub.github.io/cnn-explainer/)
- **Implementaciones**: PyTorch/TensorFlow incluyen AlexNet pre-entrenado
- **Video**: "AlexNet explained" - Yannic Kilcher (YouTube)

### El Legado 

AlexNet demostró tres verdades que definieron la década siguiente:

1. **Las redes profundas funcionan** (si tienes suficientes datos)
2. **Las GPUs son esenciales** para entrenar modelos grandes
3. **Los algoritmos de 1980-1990 (CNNs, backprop) solo necesitaban escala**

Sin AlexNet, no tendríamos: reconocimiento facial en smartphones, filtros de Instagram, autos autónomos, ni la explosión de deep learning que llevó a GPT y DALL-E.
`,
  },
  {
    id: "word2vec-2013",
    year: 2013,
    era: "Deep Learning Era",
    type: "milestone",
    title: "Word2Vec",
    description:
      "Google enseña a las máquinas que las palabras tienen significado geométrico.",
    historical_context:
      'Permitió decir matemáticamente que "Rey - Hombre + Mujer = Reina".',
    legacy: "Base de todos los embeddings modernos.",
    theory: {
      concept: "Embeddings Semánticos.",
      math: "P(w_o | w_i)",
      analogy: "Un mapa donde palabras similares viven cerca.",
    },
    practice: {
      challenge: "¿Cómo se llama la técnica de convertir palabras en vectores?",
      starter_code: "",
      expected_logic: "embeddings",
    },
    resources: ["https://arxiv.org/abs/1301.3781"],
    deep_dive: `
## Cuando las Palabras se Volvieron Números con Significado

En 2013, **Tomas Mikolov** y su equipo en Google publicaron dos papers revolucionarios que cambiaron fundamentalmente cómo las máquinas entienden el lenguaje: "Efficient Estimation of Word Representations in Vector Space" y "Distributed Representations of Words and Phrases".

Su invención, **Word2Vec**, demostró algo mágico: **las palabras tienen geometría**. Las relaciones semánticas entre palabras se pueden capturar mediante simple aritmética vectorial.

### El Problema: ¿Cómo Representar Palabras?

Antes de Word2Vec, las palabras se representaban como **one-hot encoding**:

\`\`\`
"rey"   = [1, 0, 0, 0, 0, ..., 0]  # 1 en posición 5234
"reina" = [0, 1, 0, 0, 0, ..., 0]  # 1 en posición 8921
"hombre"= [0, 0, 1, 0, 0, ..., 0]  # 1 en posición 1247
\`\`\`

**Problemas**:
- Vectores gigantes (tamaño = vocabulario, ~100K dimensiones)
- Todas las palabras están **igualmente distantes** entre sí
- No hay noción de similitud semántica

### La Idea: Embeddings Densos

Word2Vec convierte palabras en vectores **densos** de baja dimensión (~300D) donde **palabras similares están cerca**:

\`\`\`
"rey"   = [0.2, -0.4, 0.7, ..., 0.1]  # 300 dimensiones
"reina" = [0.3, -0.3, 0.6, ..., 0.2]  # Similar a "rey"
"perro" = [-0.8, 0.5, -0.2, ..., 0.9] # Lejos de "rey"
\`\`\`

**Cómo se aprenden**: Entrenando una red neuronal simple para predecir palabras vecinas en textos grandes.

### Las Dos Arquitecturas

Word2Vec tiene dos variantes:

#### 1. **Skip-gram**: Predecir contexto dado una palabra

**Objetivo**: Dada una palabra central, predecir las palabras a su alrededor

\`\`\`
Input:  "rey"
Output: [probabilidad("el"), probabilidad("de"), probabilidad("France"), ...]
\`\`\`

**Ejemplo de entrenamiento**:
- Texto: "El rey de Francia vive en París"
- Palabra central: "rey"
- Contexto (ventana de ±2): ["El", "de", "Francia", "vive"]
- Objetivo: Maximizar P(contexto | "rey")

#### 2. **CBOW** (Continuous Bag of Words): Predecir palabra dado contexto

**Objetivo**: Dado el contexto, predecir la palabra central

\`\`\`
Input:  ["El", "de", "Francia", "vive"]
Output: probabilidad("rey")
\`\`\`

**Diferencias**:
- **Skip-gram**: Mejor con datos pequeños, captura relaciones raras
- **CBOW**: Más rápido, mejor con datos grandes

### Negative Sampling: El Truco de Velocidad

Entrenar Word2Vec predecir todas las palabras del vocabulario (100K+) es lento.

**Solución**: En lugar de calcular probabilidad sobre todo el vocabulario, solo diferencia palabras positivas (que aparecen) de negativas (muestreadas al azar).

**Ejemplo**:
- Par positivo: ("rey", "reina") → Label: 1
- Pares negativos: ("rey", "pizza"), ("rey", "computadora") → Label: 0

Esto reduce el cómputo de O(vocabulario) a O(~5-20 palabras negativas).

### La Magia: Álgebra Semántica

El descubrimiento más sorprendente de Word2Vec:

**Relaciones semánticas = Operaciones vectoriales**

\`\`\`
vector("Rey") - vector("Hombre") + vector("Mujer") ≈ vector("Reina")

vector("París") - vector("Francia") + vector("Italia") ≈ vector("Roma")

vector("caminar") - vector("caminó") + vector("nadar") ≈ vector("nadó")
\`\`\`

**¿Por qué funciona?**

Los embeddings capturan **direcciones semánticas**:
- La dirección de "Hombre" → "Mujer" representa "género"
- La dirección de "Rey" → "Reina" también representa "género"
- Por lo tanto, estas direcciones son paralelas en el espacio vectorial

**Analogías que Word2Vec puede resolver**:
- Capitales: Londres : Inglaterra :: París : ?  → Francia
- Plural: gato : gatos :: perro : ?  → perros  
- Género: actor : actriz :: rey : ?  → reina
- Verbos: go : went :: take : ?  → took

### Aplicaciones y Éxito

Word2Vec se convirtió en el **estándar** de NLP (2013-2018):

**Tareas mejoradas**:
- **Clasificación de texto**: Sentiment analysis, spam detection
- **Traducción automática**: Inicializar modelos seq2seq
- **Búsqueda semántica**: Encontrar documentos similares
- **Recomendación**: "Usuarios que compraron X también compraron Y"

**Datasets populares**:
- Google News (100B palabras) → vectores pre-entrenados de 300D
- Wikipedia, Common Crawl

### Limitaciones

A pesar de su éxito, Word2Vec tiene problemas:

**1. Polisemia** (múltiples significados):
- "banco" (institución financiera) vs "banco" (asiento)
- Word2Vec da UN SOLO vector para ambos

**2. Palabras fuera de vocabulario**:
- No puede manejar palabras nuevas o con errores de ortografía

**3. No captura sintaxis**:
- Orden de palabras importante ("perro muerde hombre" ≠ "hombre muerde perro")

**4. Contexto fijo**:
- El significado no cambia según la oración

### Evolución: De Word2Vec a Transformers

Word2Vec inspiró una familia de embeddings:

**2014 - GloVe** (Stanford): Embeddings usando estadísticas globales
**2018 - ELMo**: Embeddings contextuales (significado depende de la oración)
**2018 - BERT**: Embeddings bidirectionales con Transformers
**2023 - GPT**: Ya no se usan embeddings estáticos, todo es contextual

**Legado**: Word2Vec demostró que:
✅ El significado se puede capturar geométricamente
✅ Entrenar en textos masivos aprende conocimiento del mundo
✅ Transfer learning funciona en NLP (usar embeddings pre-entrenados)

Todos los LLMs modernos (GPT-4, Claude) usan la misma idea de embeddings, pero **contextuales** (el vector de "banco" cambia según la oración).

## Implementación Conceptual

\`\`\`python
import numpy as np
from collections import defaultdict

class SimpleWord2Vec:
    """
    Implementación simplificada de Word2Vec (Skip-gram)
    Para fines educativos - la versión real usa optimizaciones complejas
    """
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
        
    def build_vocab(self, sentences):
        """Construir vocabulario a partir de oraciones"""
        vocab = set()
        for sentence in sentences:
            vocab.update(sentence.lower().split())
        
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(vocab))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Inicializar embeddings aleatorios
        vocab_size = len(self.word_to_idx)
        self.embeddings = np.random.randn(vocab_size, self.embedding_dim) * 0.01
        
    def get_training_pairs(self, sentence, window_size=2):
        """Generar pares (palabra_central, palabra_contexto) para Skip-gram"""
        words = sentence.lower().split()
        pairs = []
        
        for i, target_word in enumerate(words):
            # Palabras de contexto dentro de la ventana
            for j in range(max(0, i - window_size), 
                          min(len(words), i + window_size + 1)):
                if i != j:
                    context_word = words[j]
                    pairs.append((target_word, context_word))
        
        return pairs
    
    def similarity(self, word1, word2):
        """Calcular similitud coseno entre dos palabras"""
        if word1 not in self.word_to_idx or word2 not in self.word_to_idx:
            return 0.0
        
        idx1 = self.word_to_idx[word1]
        idx2 = self.word_to_idx[word2]
        
        vec1 = self.embeddings[idx1]
        vec2 = self.embeddings[idx2]
        
        # Similitud coseno
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    def most_similar(self, word, top_n=5):
        """Encontrar las palabras más similares"""
        if word not in self.word_to_idx:
            return []
        
        word_idx = self.word_to_idx[word]
        word_vec = self.embeddings[word_idx]
        
        # Calcular similitud con todas las palabras
        similarities = []
        for idx, other_word in self.idx_to_word.items():
            if idx != word_idx:
                other_vec = self.embeddings[idx]
                sim = np.dot(word_vec, other_vec) / (
                    np.linalg.norm(word_vec) * np.linalg.norm(other_vec)
                )
                similarities.append((other_word, sim))
        
        # Ordenar y retornar top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    
    def analogy(self, word_a, word_b, word_c):
        """Resolver analogías: A es a B como C es a ?"""
        # word_a - word_b + word_c ≈ ?
        if not all(w in self.word_to_idx for w in [word_a, word_b, word_c]):
            return None
        
        vec_a = self.embeddings[self.word_to_idx[word_a]]
        vec_b = self.embeddings[self.word_to_idx[word_b]]
        vec_c = self.embeddings[self.word_to_idx[word_c]]
        
        # Operación vectorial
        target_vec = vec_a - vec_b + vec_c
        
        # Encontrar palabra más cercana
        max_sim = -1
        result_word = None
        
        for word, idx in self.word_to_idx.items():
            if word in [word_a, word_b, word_c]:
                continue
            
            vec = self.embeddings[idx]
            sim = np.dot(target_vec, vec) / (
                np.linalg.norm(target_vec) * np.linalg.norm(vec)
            )
            
            if sim > max_sim:
                max_sim = sim
                result_word = word
        
        return result_word

# Ejemplo de uso
sentences = [
    "El rey vive en el castillo",
    "La reina vive en el palacio",
    "El hombre camina por la calle",
    "La mujer camina por el parque",
]

model = SimpleWord2Vec(embedding_dim=50)
model.build_vocab(sentences)

print("Vocabulario:", len(model.word_to_idx), "palabras")
print("\\nEjemplo de analogía (después de entrenar):")
print("rey - hombre + mujer ≈", model.analogy("rey", "hombre", "mujer"))
\`\`\`

**Nota**: Esta es una versión ultra-simplificada para ilustrar los conceptos. Word2Vec real usa:
- Negative sampling para eficiencia
- Subsampling de palabras frecuentes
- Optimización con SGD
- Vocabulario de millones de palabras

## Recursos para Profundizar

- **Paper original**: Mikolov, T. et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- **Tutorial interactivo**: TensorFlow Word2Vec tutorial
- **Visualización**: Projector.tensorflow.org - Visualiza embeddings en 3D
- **Pre-trained vectors**: Google News Word2Vec (3M palabras, 300D)
    `,
  },
  {
    id: "gans-2014",
    year: 2014,
    era: "Deep Learning Era",
    type: "milestone",
    title: "Redes GANs",
    description: 'Ian Goodfellow inventa redes que pueden "imaginar".',
    historical_context: "Dos redes compitiendo: generador vs discriminador.",
    legacy: "Inicio del contenido sintético (Deepfakes).",
    theory: {
      concept: "Minimax Game.",
      math: "V(D,G)",
      analogy: "Un falsificador tratando de engañar a un experto de arte.",
    },
    practice: {
      challenge: "¿Cómo se llama la red que intenta detectar si es falso?",
      starter_code: "",
      expected_logic: "discriminador",
    },
  },
  {
    id: "resnet-2015",
    year: 2015,
    era: "Deep Learning Era",
    type: "milestone",
    title: "ResNet",
    description: "Microsoft permite redes de cientos de capas.",
    historical_context: "Las redes profundas antes dejaban de aprender.",
    legacy: "Permitió la profundidad extrema.",
    theory: {
      concept: "Skip Connections.",
      math: "y = F(x) + x",
      analogy:
        "Puentes que permiten que la información salte partes del cerebro.",
    },
    practice: {
      challenge: "¿Cómo se llaman las conexiones que saltan capas?",
      starter_code: "",
      expected_logic: "residuales",
    },
  },
  {
    id: "transformers-2017",
    year: 2017,
    era: "Generative Era",
    type: "era-change",
    title: "Transformers",
    description: 'Google publica "Attention Is All You Need".',
    historical_context: "El nacimiento de la arquitectura GPT.",
    legacy: "La base de toda la IA moderna.",
    theory: {
      concept: "Self-Attention.",
      math: "\\text{softmax}(\\frac{QK^T}{sqrt{d_k}})V$",
      analogy: "Subrayar lo importante de una página de un vistazo.",
    },
    practice: {
      challenge: "¿Mecanismo clave de los Transformers?",
      starter_code: "",
      expected_logic: "atención",
    },
    deep_dive: `## La Arquitectura que Cambió Todo

En 2017, un equipo de Google publicó el paper "**Attention Is All You Need**" que revolucionó el procesamiento de lenguaje natural y, eventualmente, toda la inteligencia artificial. Este paper introdujo la arquitectura **Transformer**, que es la base de GPT, BERT, ChatGPT, DALL-E, y practicamente todos los modelos de IA modernos.

### El Problema que Resolvieron

Antes de los Transformers, el procesamiento de lenguaje usaba **Redes Neuronales Recurrentes (RNNs)** y **LSTMs**. Estos modelos procesaban texto **secuencialmente** (palabra por palabra), lo que tenía limitaciones:

1. **Lento**: No se podía paralelizar (cada palabra dependía de la anterior)
2. **Memoria limitada**: Dificultad para recordar información de textos largos
3. **Gradientes que desaparecen**: Problemas de entrenamiento en secuencias largas

Los Transformers resuelven esto procesando **todo el texto en paralelo** usando **mecanismos de atención**.

### ¿Qué es Self-Attention?

**Self-Attention** permite que cada palabra "mire" a todas las demás palabras de la secuencia y decida cuáles son importantes para entenderla.

**Ejemplo**: En la frase "El animal no cruzó la calle porque **estaba** demasiado cansado"

¿A qué se refiere "estaba"? ¿Al animal o a la calle?

Self-attention calcula scores de relevancia:
- "estaba" → "animal": **0.85** (alta relevancia) ✓
- "estaba" → "calle": **0.12** (baja relevancia)

¡El modelo entiende que"estaba" se refiere al "animal"!

### La Fórmula Matemática

La atención se calcula con tres matrices aprendidas: **Q** (Query), **K** (Key), **V** (Value)

\`\`\`
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Donde:
- Q = Queries (lo que estoy buscando)
- K = Keys (índice de contenido)
- V =  Values (el contenido real)
- d_k = dimensión de las keys (para estabilidad numérica)
\`\`\`

**Paso a paso**:
1. Calcula similitud entre queries y keys: \`QK^T\`
2. Escala por \`√d_k\` para evitar valores muy grandes
3. Aplica softmax para obtener probabilidades
4. Usa estas probabilidades para ponderar los values

### Arquitectura del Transformer

\`\`\`
Input → [Embedding + Positional Encoding]
           ↓
       [Encoder Stack]
       - Multi-Head Self-Attention
       - Feed Forward Network
       × N capas (ej: 12)
           ↓
       [Decoder Stack]
       - Masked Self-Attention
       - Encoder-Decoder Attention
       - Feed Forward Network
       × N capas
           ↓
        Output
\`\`\`

**Multi-Head Attention**: En lugar de una atención, usa múltiples "cabezas" en paralelo, cada una aprendiendo diferentes tipos de relaciones (sintaxis, semántica, etc.)

### Código Simplificado en PyTorch

\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        
        # Matrices para Query, Key, Value
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # QK^T / sqrt(d_k)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Softmax para obtener attention weights
        attention = F.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # Multiply by values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        
        return self.fc_out(out)

# Uso
model = SelfAttention(embed_size=512, heads=8)
x = torch.randn(32, 10, 512)  # (batch, seq_len, embed_dim)
output = model(x, x, x, mask=None)
print(output.shape)  # → torch.Size([32, 10, 512])
\`\`\`

### Por Qué es Revolucionario

1. **Paralelización**: Procesa todo el texto a la vez (vs RNNs secuenciales)
2. **Dependencias a largo plazo**: Puede relacionar palabras muy distantes
3. **Escalabilidad**: Funciona mejor con más datos y más parámetros
4. **Versatilidad**: Funciona para texto, imágenes, audio, video

### El Impacto

Los Transformers desencadenaron una explosión de modelos:

**NLP (Lenguaje)**:
- BERT (2018): Comprensión bidireccional
- GPT-2/3/4 (2019-2023): Generación de texto
- ChatGPT (2022): Conversación + RLHF
- Claude, Gemini, LLaMA...

**Visión**:
- Vision Transformer (ViT): Transformers para imágenes
- DALL-E, Stable Diffusion: Generación de imágenes

**Multimodal**:
- CLIP: Conecta texto e imágenes
- GPT-4V: Entiende imágenes y texto
- Sora: Generación de video

**Ciencia**:
- AlphaFold 2: Plegamiento de proteínas con atención

### Recursos para Profundizar

- **Paper original**: "Attention Is All You Need" (Vaswani et al., 2017) [arxiv.org/abs/1706.03762]
- **Video explicativo**: "Attention is all you need" - Yannic Kilcher (YouTube)
- **Post visual**: "The Illustrated Transformer" - Jay Alammar (jalammar.github.io)
- **Implementación completa**: "The Annotated Transformer" (Harvard NLP)
- **Curso**: "CS224N: NLP with Deep Learning" - Stanford (web.stanford.edu/class/cs224n/)

### El Legado

El paper de Transformers tiene más de **80,000 citas** (2024) y es uno de los papers más influyentes de la historia de la IA. Sin Transformers, no tendríamos ChatGPT, ni la revolución de la IA generativa que estamos viviendo hoy.
`,
  },
  {
    id: "alphafold-2021",
    year: 2021,
    era: "Generative Era",
    type: "milestone",
    title: "AlphaFold 2",
    description: "DeepMind resuelve el plegamiento de proteínas.",
    historical_context: "IA aplicada a la biología para acelerar la medicina.",
    legacy: "Considerado el mayor aporte de la IA a la ciencia.",
    theory: {
      concept: "Geometric Deep Learning.",
      math: "Evoformer Architecture",
      analogy: "Un experto en origami molecular.",
    },
    practice: {
      challenge: "¿Qué problema biológico resolvió?",
      starter_code: "",
      expected_logic: "plegamiento de proteínas",
    },
  },
  {
    id: "diffusion-2022",
    year: 2022,
    era: "Generative Era",
    type: "milestone",
    title: "Modelos de Difusión",
    description: "Imágenes desde el ruido (Midjourney/DALL-E).",
    historical_context: "Creación de arte realista.",
    legacy: "Revolución en la generación visual.",
    theory: {
      concept: "Reverse Diffusion.",
      math: "U-Net Denoising",
      analogy: "Esculpir una estatua perfecta desde estática de TV.",
    },
    practice: {
      challenge: "¿De qué base parten estos modelos?",
      starter_code: "",
      expected_logic: "ruido",
    },
  },
  {
    id: "chatgpt-2022",
    year: 2022,
    era: "Generative Era",
    type: "milestone",
    title: "ChatGPT",
    description:
      "OpenAI lanza un chatbot que convence a millones de que la IA 'entiende'.",
    historical_context:
      "Combina GPT-3.5 con Aprendizaje por Refuerzo basado en Retroalimentación Humana.",
    legacy: "El momento en que la IA se volvió mainstream.",
    theory: {
      concept: "RLHF (Reinforcement Learning from Human Feedback).",
      math: "Policy Gradient",
      analogy:
        "Un estudiante que mejora basándose en las calificaciones de sus profesores.",
    },
    practice: {
      challenge: "¿Qué técnica usa ChatGPT para alinearse con valores humanos?",
      starter_code: "",
      expected_logic: "RLHF",
    },
  },
  {
    id: "lora-2023",
    year: 2023,
    era: "Generative Era",
    type: "milestone",
    title: "LoRA",
    description: "Ajuste fino de modelos gigantes con pocos recursos.",
    historical_context:
      "Permitió que personas normales entrenen modelos como Llama.",
    legacy: "Democratización del fine-tuning.",
    theory: {
      concept: "Low-Rank Adaptation.",
      math: "W = W_0 + AB",
      analogy:
        "Solo cambiar unos pocos tornillos en un motor gigante para que corra distinto.",
    },
    practice: {
      challenge: "¿Qué significan las siglas LoRA?",
      starter_code: "",
      expected_logic: "Low-Rank Adaptation",
    },
  },
  {
    id: "moe-2023",
    year: 2023,
    era: "Generative Era",
    type: "milestone",
    title: "Mixture of Experts (MoE)",
    description:
      "Modelos que solo usan una parte de su cerebro para cada tarea.",
    historical_context:
      "Mixtral 8x7B demuestra que MoE supera a modelos densos.",
    legacy: "Escalabilidad eficiente hacia trillones de parámetros.",
    theory: {
      concept: "Sparse Activation.",
      math: "Gating Network",
      analogy:
        "Una oficina llena de expertos donde solo el traductor trabaja si el texto es en francés.",
    },
    practice: {
      challenge: "¿Cómo se llama la red que elige qué experto usar?",
      starter_code: "",
      expected_logic: "gating",
    },
  },
  {
    id: "sora-2024",
    year: 2024,
    era: "Era of Reasoning",
    type: "milestone",
    title: "Sora y Video",
    description: "Generación de mundos simulados con física coherente.",
    historical_context: "Transición a Modelos del Mundo.",
    legacy: "Hacia la IA que entiende la realidad física.",
    theory: {
      concept: "Spatiotemporal Patches.",
      math: "Diffusion Transformers (DiT)",
      analogy: "Un director de cine que puede soñar películas enteras.",
    },
    practice: {
      challenge: "¿Qué patches usa Sora?",
      starter_code: "",
      expected_logic: "espaciotemporales",
    },
  },
  {
    id: "reasoning-2025",
    year: 2025,
    era: "Era of Reasoning",
    type: "milestone",
    title: "Razonamiento System 2",
    description: "IAs que piensan antes de hablar (Modelos o1).",
    historical_context: "Uso masivo de Chain of Thought oculto.",
    legacy:
      "La IA se vuelve un compañero de resolución de problemas complejos.",
    theory: {
      concept: "Test-Time Compute.",
      math: "Search-based Reasoning",
      analogy:
        "Un experto que escribe borradores y se autocorrige antes de responder.",
    },
    practice: {
      challenge: "¿Cómo se llama la técnica de pensar paso a paso?",
      starter_code: "",
      expected_logic: "cadena de pensamiento",
    },
  },
];
