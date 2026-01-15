import { useState, useEffect } from "react";

/**
 * Hook para manejar el progreso del usuario en la línea de tiempo de IA
 * Usa localStorage para persistencia local
 */

export interface UserProgress {
  completedMilestones: string[];
  inProgressMilestones: string[];
  currentMilestone?: string;
  currentPath?: string;
  timeSpent: number; // en minutos
  lastVisit: string; // ISO date string
  achievements: string[];
  startedDate: string; // ISO date string
}

const STORAGE_KEY = "nexus-ai-user-progress";

const DEFAULT_PROGRESS: UserProgress = {
  completedMilestones: [],
  inProgressMilestones: [],
  currentMilestone: undefined,
  currentPath: undefined,
  timeSpent: 0,
  lastVisit: new Date().toISOString(),
  achievements: [],
  startedDate: new Date().toISOString(),
};

export function useUserProgress() {
  const [progress, setProgress] = useState<UserProgress>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (error) {
      console.error("Error loading progress:", error);
    }
    return DEFAULT_PROGRESS;
  });

  // Guardar en localStorage cada vez que cambie el progreso
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
    } catch (error) {
      console.error("Error saving progress:", error);
    }
  }, [progress]);

  // Actualizar lastVisit al montar el componente
  useEffect(() => {
    setProgress((prev) => ({
      ...prev,
      lastVisit: new Date().toISOString(),
    }));
  }, []);

  /**
   * Marca un milestone como completado
   */
  const completeMilestone = (milestoneId: string) => {
    setProgress((prev) => {
      const newCompleted = Array.from(
        new Set([...prev.completedMilestones, milestoneId])
      );
      const newInProgress = prev.inProgressMilestones.filter(
        (id) => id !== milestoneId
      );

      // Check for achievements
      const newAchievements = [...prev.achievements];
      if (
        newCompleted.length === 1 &&
        !newAchievements.includes("first-step")
      ) {
        newAchievements.push("first-step");
      }
      if (
        newCompleted.length === 5 &&
        !newAchievements.includes("bronze-learner")
      ) {
        newAchievements.push("bronze-learner");
      }
      if (
        newCompleted.length === 10 &&
        !newAchievements.includes("silver-learner")
      ) {
        newAchievements.push("silver-learner");
      }
      if (
        newCompleted.length === 22 &&
        !newAchievements.includes("master-historian")
      ) {
        newAchievements.push("master-historian");
      }

      return {
        ...prev,
        completedMilestones: newCompleted,
        inProgressMilestones: newInProgress,
        achievements: newAchievements,
      };
    });
  };

  /**
   * Marca un milestone como en progreso
   */
  const startMilestone = (milestoneId: string) => {
    setProgress((prev) => {
      // No agregar si ya está completado
      if (prev.completedMilestones.includes(milestoneId)) {
        return prev;
      }

      const newInProgress = Array.from(
        new Set([...prev.inProgressMilestones, milestoneId])
      );

      return {
        ...prev,
        inProgressMilestones: newInProgress,
        currentMilestone: milestoneId,
      };
    });
  };

  /**
   * Establece la ruta de aprendizaje actual
   */
  const setCurrentPath = (pathId: string) => {
    setProgress((prev) => ({
      ...prev,
      currentPath: pathId,
    }));
  };

  /**
   * Agrega tiempo de estudio (en minutos)
   */
  const addStudyTime = (minutes: number) => {
    setProgress((prev) => ({
      ...prev,
      timeSpent: prev.timeSpent + minutes,
    }));
  };

  /**
   * Reinicia el progreso (útil para testing)
   */
  const resetProgress = () => {
    setProgress(DEFAULT_PROGRESS);
    localStorage.removeItem(STORAGE_KEY);
  };

  /**
   * Calcula el porcentaje de progreso general
   */
  const getProgressPercentage = (totalMilestones: number = 22): number => {
    return Math.round(
      (progress.completedMilestones.length / totalMilestones) * 100
    );
  };

  /**
   * Verifica si un milestone está completado
   */
  const isMilestoneCompleted = (milestoneId: string): boolean => {
    return progress.completedMilestones.includes(milestoneId);
  };

  /**
   * Verifica si un milestone está en progreso
   */
  const isMilestoneInProgress = (milestoneId: string): boolean => {
    return progress.inProgressMilestones.includes(milestoneId);
  };

  return {
    progress,
    completeMilestone,
    startMilestone,
    setCurrentPath,
    addStudyTime,
    resetProgress,
    getProgressPercentage,
    isMilestoneCompleted,
    isMilestoneInProgress,
  };
}
