

export type MilestoneType = 'milestone' | 'era-change' | 'crisis';

export interface Milestone {
  id: string;
  year: number;
  title: string;
  era: string;
  type: MilestoneType;
  description: string;
  historical_context: string;
  legacy: string;
  theory: {
    concept: string;
    math: string;
    analogy: string;
  };
  practice: {
    challenge: string;
    starter_code: string;
    expected_logic: string;
  };
  deep_dive?: string; 
}

export interface UserState {
  masteredIds: string[];
  currentId: string | null;
}

export type Category = 
  | 'math-foundation' 
  | 'statistical-mechanics' 
  | 'neural-architectures' 
  | 'optimization-theory' 
  | 'data-engineering' 
  | 'deployment-scaling';

/**
 * Represents the seniority levels of the user within the Nexus ecosystem.
 */
export type UserRank = 'novice' | 'architect' | 'master' | 'visionary';

export interface AIElement {
  symbol: string;
  name: string;
  category: Category;
  atomic_number: number;
  position: { x: number; y: number };
  prerequisites: { symbol: string }[];
}