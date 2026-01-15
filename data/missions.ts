import { Category } from '../types';

export interface Mission {
  id: string;
  title: string;
  description: string;
  unlocksCategory: Category;
  requiredMasteryCount: number;
}

// Defining progression missions to unlock new knowledge categories
export const NEXUS_MISSIONS: Mission[] = [
  {
    id: 'm1',
    title: 'Mathematical Genesis',
    description: 'Master the basics of linear algebra and calculus to establish your core foundation.',
    unlocksCategory: 'math-foundation',
    requiredMasteryCount: 0
  },
  {
    id: 'm2',
    title: 'The Uncertainty Principle',
    description: 'Master 2 concepts to unlock the Statistical Mechanics layer.',
    unlocksCategory: 'statistical-mechanics',
    requiredMasteryCount: 2
  },
  {
    id: 'm3',
    title: 'Neural Synthesis',
    description: 'Master 4 concepts to unlock Neural Architectures and deep learning structures.',
    unlocksCategory: 'neural-architectures',
    requiredMasteryCount: 4
  }
];

export const MISSIONS = NEXUS_MISSIONS;
