
import { useState, useEffect, useMemo } from 'react';
import { UserState } from '../types';
import { AI_TIMELINE } from '../data/ai_timeline';

export const useNexusStore = () => {
  const [state, setState] = useState<UserState>(() => {
    const saved = localStorage.getItem('nexus_ai_v80_progress');
    return saved ? JSON.parse(saved) : { masteredIds: [], currentId: null };
  });

  useEffect(() => {
    localStorage.setItem('nexus_ai_v80_progress', JSON.stringify(state));
  }, [state]);

  const progress = useMemo(() => {
    return Math.round((state.masteredIds.length / AI_TIMELINE.length) * 100);
  }, [state.masteredIds]);

  const markAsMastered = (id: string) => {
    if (!state.masteredIds.includes(id)) {
      setState(prev => ({
        ...prev,
        masteredIds: [...prev.masteredIds, id]
      }));
    }
  };

  return { state, markAsMastered, progress };
};
