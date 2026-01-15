import { AIElement, Category } from '../types';

// Visual themes and metadata for each knowledge category
export const CATEGORY_THEMES: Record<Category, { border: string; color: string; desc: string; glow: string }> = {
  'math-foundation': {
    border: 'border-blue-500/50',
    color: 'text-blue-400',
    desc: 'The essential mathematical building blocks of artificial intelligence.',
    glow: 'shadow-[0_0_15px_rgba(59,130,246,0.3)]'
  },
  'statistical-mechanics': {
    border: 'border-emerald-500/50',
    color: 'text-emerald-400',
    desc: 'Probabilistic models and statistical inference techniques.',
    glow: 'shadow-[0_0_15px_rgba(16,185,129,0.3)]'
  },
  'neural-architectures': {
    border: 'border-purple-500/50',
    color: 'text-purple-400',
    desc: 'Structure and design of modern neural networks.',
    glow: 'shadow-[0_0_15px_rgba(168,85,247,0.3)]'
  },
  'optimization-theory': {
    border: 'border-amber-500/50',
    color: 'text-amber-400',
    desc: 'Algorithms for training and refining model parameters.',
    glow: 'shadow-[0_0_15px_rgba(245,158,11,0.3)]'
  },
  'data-engineering': {
    border: 'border-rose-500/50',
    color: 'text-rose-400',
    desc: 'Processing and managing large-scale information systems.',
    glow: 'shadow-[0_0_15px_rgba(244,63,94,0.3)]'
  },
  'deployment-scaling': {
    border: 'border-cyan-500/50',
    color: 'text-cyan-400',
    desc: 'Strategies for serving and scaling AI models in production.',
    glow: 'shadow-[0_0_15px_rgba(6,182,212,0.3)]'
  }
};

// List of all knowledge elements following a periodic table layout
export const AI_ELEMENTS: AIElement[] = [
  { symbol: 'Li', name: 'Linear Algebra', category: 'math-foundation', atomic_number: 1, position: { x: 1, y: 1 }, prerequisites: [] },
  { symbol: 'Ca', name: 'Calculus', category: 'math-foundation', atomic_number: 2, position: { x: 2, y: 1 }, prerequisites: [] },
  { symbol: 'Pr', name: 'Probability', category: 'statistical-mechanics', atomic_number: 3, position: { x: 1, y: 2 }, prerequisites: [{ symbol: 'Ca' }] },
  { symbol: 'St', name: 'Statistics', category: 'statistical-mechanics', atomic_number: 4, position: { x: 2, y: 2 }, prerequisites: [{ symbol: 'Pr' }] },
  { symbol: 'Pe', name: 'Perceptron', category: 'neural-architectures', atomic_number: 5, position: { x: 3, y: 1 }, prerequisites: [{ symbol: 'Li' }] },
  { symbol: 'Cn', name: 'CNN', category: 'neural-architectures', atomic_number: 6, position: { x: 4, y: 1 }, prerequisites: [{ symbol: 'Pe' }] },
  { symbol: 'Gd', name: 'Gradient Descent', category: 'optimization-theory', atomic_number: 7, position: { x: 3, y: 2 }, prerequisites: [{ symbol: 'Ca' }] },
  { symbol: 'Bp', name: 'Backpropagation', category: 'optimization-theory', atomic_number: 8, position: { x: 4, y: 2 }, prerequisites: [{ symbol: 'Gd' }] },
  { symbol: 'Et', name: 'ETL Pipelines', category: 'data-engineering', atomic_number: 9, position: { x: 18, y: 1 }, prerequisites: [] },
  { symbol: 'Do', name: 'Docker', category: 'deployment-scaling', atomic_number: 10, position: { x: 18, y: 7 }, prerequisites: [{ symbol: 'Et' }] },
];

export const ELEMENTS = AI_ELEMENTS;
