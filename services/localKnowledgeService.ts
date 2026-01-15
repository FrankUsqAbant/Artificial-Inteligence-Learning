/**
 * Servicio de Conocimiento Local
 * Sistema de búsqueda y respuesta basado en datos locales
 * Sin dependencia de APIs externas
 */

import { AI_TIMELINE } from "../data/ai_timeline";
import { FAQ_DATABASE } from "../data/faq_database";
import { AI_GLOSSARY } from "../data/glossary";
import { FUNDAMENTAL_CONCEPTS } from "../data/fundamental_concepts";
import type { Milestone } from "../types";

/**
 * Sistema de Conocimiento Local para la Línea de Tiempo de IA
 * Reemplaza la dependencia de Google Gemini API con búsqueda local inteligente
 */

export interface SearchResult {
  type: "milestone" | "faq" | "glossary";
  relevance: number;
  data: any;
  snippet: string;
}

export interface Answer {
  text: string;
  sources: string[];
  relatedMilestones: Milestone[];
}

/**
 * Calcula la similitud entre dos strings usando algoritmo simple
 * Retorna un valor entre 0 (sin similitud) y 1 (idéntico)
 */
function calculateSimilarity(str1: string, str2: string): number {
  const s1 = str1.toLowerCase();
  const s2 = str2.toLowerCase();

  // Coincidencia exacta
  if (s1 === s2) return 1;

  // Contiene la palabra completa
  if (s1.includes(s2) || s2.includes(s1)) return 0.8;

  // Coincidencia de palabras
  const words1 = s1.split(/\s+/);
  const words2 = s2.split(/\s+/);

  let matches = 0;
  words1.forEach((w1) => {
    words2.forEach((w2) => {
      if (w1 === w2 || w1.includes(w2) || w2.includes(w1)) {
        matches++;
      }
    });
  });

  const maxWords = Math.max(words1.length, words2.length);
  return maxWords > 0 ? matches / maxWords : 0;
}

/**
 * Busca en todos los milestones por palabras clave
 */
export function searchKnowledge(query: string): SearchResult[] {
  const results: SearchResult[] = [];
  const queryLower = query.toLowerCase();

  // Buscar en milestones
  AI_TIMELINE.forEach((milestone) => {
    let relevance = 0;

    // Buscar en título
    relevance += calculateSimilarity(queryLower, milestone.title) * 3;

    // Buscar en descripción
    relevance += calculateSimilarity(queryLower, milestone.description) * 2;

    // Buscar en concepto teórico
    relevance += calculateSimilarity(queryLower, milestone.theory.concept);

    // Buscar en contexto histórico
    if (milestone.historical_context.toLowerCase().includes(queryLower)) {
      relevance += 0.5;
    }

    if (relevance > 0.3) {
      results.push({
        type: "milestone",
        relevance,
        data: milestone,
        snippet: milestone.description.substring(0, 150) + "...",
      });
    }
  });

  // Buscar en FAQ
  FAQ_DATABASE.forEach((faq) => {
    let relevance = 0;

    relevance += calculateSimilarity(queryLower, faq.question) * 2;

    faq.keywords.forEach((keyword) => {
      if (queryLower.includes(keyword)) {
        relevance += 0.8;
      }
    });

    if (relevance > 0.3) {
      results.push({
        type: "faq",
        relevance,
        data: faq,
        snippet: faq.answer.substring(0, 150) + "...",
      });
    }
  });

  // Buscar en glosario
  Object.entries(AI_GLOSSARY).forEach(([key, entry]) => {
    let relevance = 0;

    relevance += calculateSimilarity(queryLower, (entry as any).term) * 2;
    relevance += calculateSimilarity(queryLower, (entry as any).definition);

    if (relevance > 0.3) {
      results.push({
        type: "glossary",
        relevance,
        data: entry,
        snippet: (entry as any).definition.substring(0, 150) + "...",
      });
    }
  });

  // Buscar en conceptos fundamentales
  Object.values(FUNDAMENTAL_CONCEPTS).forEach((concept) => {
    let relevance = 0;

    // Buscar en término
    relevance += calculateSimilarity(queryLower, concept.term) * 3;

    // Buscar en definición corta
    relevance += calculateSimilarity(queryLower, concept.shortDefinition) * 2;

    // Buscar en explicación detallada
    relevance += calculateSimilarity(queryLower, concept.detailedExplanation);

    // Bonus si coincide exactamente con el ID
    if (concept.id.includes(queryLower.replace(/\s+/g, "-"))) {
      relevance += 0.8;
    }

    if (relevance > 0.3) {
      results.push({
        type: "concept" as any,
        relevance,
        data: concept,
        snippet: concept.shortDefinition,
      });
    }
  });

  // Ordenar por relevancia
  return results.sort((a, b) => b.relevance - a.relevance);
}

/**
 * Encuentra milestones relacionados a uno dado
 */
export function findRelatedMilestones(milestoneId: string): Milestone[] {
  const milestone = AI_TIMELINE.find((m) => m.id === milestoneId);
  if (!milestone) return [];

  // Buscar milestones de la misma era
  const sameEra = AI_TIMELINE.filter(
    (m) => m.era === milestone.era && m.id !== milestoneId
  );

  // Buscar milestones con conceptos similares
  const similarConcepts = AI_TIMELINE.filter((m) => {
    if (m.id === milestoneId) return false;
    return (
      calculateSimilarity(m.theory.concept, milestone.theory.concept) > 0.3
    );
  });

  // Combinar y eliminar duplicados
  const related = [...sameEra, ...similarConcepts];
  const unique = Array.from(new Map(related.map((m) => [m.id, m])).values());

  return unique.slice(0, 5); // Máximo 5 relacionados
}

/**
 * Busca en el glosario
 */
export function searchGlossary(
  term: string
): (typeof AI_GLOSSARY)[keyof typeof AI_GLOSSARY] | null {
  const termLower = term.toLowerCase();

  // Búsqueda exacta
  if (AI_GLOSSARY[termLower]) {
    return AI_GLOSSARY[termLower];
  }

  // Búsqueda parcial
  for (const [key, entry] of Object.entries(AI_GLOSSARY)) {
    if (key.includes(termLower) || termLower.includes(key)) {
      return entry;
    }
  }

  return null;
}

/**
 * Responde una pregunta usando conocimiento local
 */
export function answerQuestion(question: string): Answer {
  const searchResults = searchKnowledge(question);

  if (searchResults.length === 0) {
    return {
      text: "No encontré información específica sobre esa pregunta en la cronología de IA. ¿Podrías reformular tu pregunta o preguntar sobre algún hito específico?",
      sources: [],
      relatedMilestones: [],
    };
  }

  const topResult = searchResults[0];
  let text = "";
  let sources: string[] = [];
  let relatedMilestones: Milestone[] = [];

  switch (topResult.type) {
    case "faq":
      text = topResult.data.answer;
      sources = ["Base de Preguntas Frecuentes"];
      relatedMilestones = topResult.data.relatedMilestones
        .map((id: string) => AI_TIMELINE.find((m) => m.id === id))
        .filter(Boolean);
      break;

    case "milestone":
      const milestone = topResult.data;
      text = `${milestone.description}\n\n**Contexto:** ${milestone.historical_context}\n\n**Concepto clave:** ${milestone.theory.concept}`;
      sources = [`${milestone.title} (${milestone.year})`];
      relatedMilestones = findRelatedMilestones(milestone.id);
      break;

    case "glossary":
      text = topResult.data.definition;
      sources = ["Glosario de Términos de IA"];
      relatedMilestones = topResult.data.relatedMilestones
        .map((id: string) => AI_TIMELINE.find((m) => m.id === id))
        .filter(Boolean);
      break;
  }

  return {
    text,
    sources,
    relatedMilestones,
  };
}

/**
 * Obtiene sugerencias de preguntas comunes
 */
export function getSuggestedQuestions(): string[] {
  return FAQ_DATABASE.slice(0, 5).map((faq) => faq.question);
}
