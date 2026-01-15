import React from "react";
import { LEARNING_PATHS, type LearningPath } from "../data/learning_paths";
import { useUserProgress } from "../hooks/useUserProgress";
import { AI_TIMELINE } from "../data/ai_timeline";

interface LearningPathVisualizerProps {
  onSelectMilestone?: (milestoneId: string) => void;
}

const LearningPathVisualizer: React.FC<LearningPathVisualizerProps> = ({
  onSelectMilestone,
}) => {
  const { progress, setCurrentPath, isMilestoneCompleted } = useUserProgress();
  const [selectedPath, setSelectedPath] = React.useState<string | null>(
    progress.currentPath || null
  );

  const handlePathSelect = (pathId: string) => {
    setSelectedPath(pathId);
    setCurrentPath(pathId);
  };

  const getPathProgress = (path: LearningPath): number => {
    const completedCount = path.milestones.filter((id) =>
      isMilestoneCompleted(id)
    ).length;
    return Math.round((completedCount / path.milestones.length) * 100);
  };

  const selectedPathData = selectedPath
    ? LEARNING_PATHS.find((p) => p.id === selectedPath)
    : null;

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-cyan-500/30 rounded-lg p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="text-2xl">🎓</div>
        <h2 className="text-2xl font-bold text-cyan-400">
          Rutas de Aprendizaje
        </h2>
      </div>

      {/* Lista de Rutas */}
      {!selectedPath && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {LEARNING_PATHS.map((path) => {
            const progressPercent = getPathProgress(path);

            return (
              <div
                key={path.id}
                onClick={() => handlePathSelect(path.id)}
                className="bg-slate-700/50 border border-cyan-500/20 rounded-lg p-4 cursor-pointer hover:border-cyan-400/50 hover:bg-slate-700/70 transition-all"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="text-xl">{path.icon || "📚"}</div>
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      path.difficulty === "beginner"
                        ? "bg-green-500/20 text-green-400"
                        : path.difficulty === "intermediate"
                        ? "bg-yellow-500/20 text-yellow-400"
                        : "bg-red-500/20 text-red-400"
                    }`}
                  >
                    {path.difficulty === "beginner"
                      ? "Principiante"
                      : path.difficulty === "intermediate"
                      ? "Intermedio"
                      : "Avanzado"}
                  </span>
                </div>

                <h3 className="text-lg font-bold text-white mb-1">
                  {path.title}
                </h3>
                <p className="text-sm text-gray-400 mb-3">{path.description}</p>

                <div className="space-y-2">
                  {/* Progress Bar */}
                  <div className="w-full bg-slate-600 rounded-full h-2">
                    <div
                      className="bg-cyan-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>

                  <div className="flex justify-between text-xs text-gray-400">
                    <span>{progressPercent}% completado</span>
                    <span>⏱️ {path.estimatedHours}h</span>
                  </div>

                  <div className="flex gap-2 text-xs text-gray-500">
                    <span>📍 {path.milestones.length} hitos</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Vista Detallada de la Ruta Seleccionada */}
      {selectedPath && selectedPathData && (
        <div>
          <button
            onClick={() => setSelectedPath(null)}
            className="mb-4 text-cyan-400 hover:text-cyan-300 flex items-center gap-2"
          >
            ← Volver a rutas
          </button>

          <div className="bg-slate-700/30 rounded-lg p-6 mb-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-2xl font-bold text-white mb-2">
                  {selectedPathData.icon} {selectedPathData.title}
                </h3>
                <p className="text-gray-400">{selectedPathData.description}</p>
              </div>
              <span
                className={`text-sm px-3 py-1 rounded ${
                  selectedPathData.difficulty === "beginner"
                    ? "bg-green-500/20 text-green-400"
                    : selectedPathData.difficulty === "intermediate"
                    ? "bg-yellow-500/20 text-yellow-400"
                    : "bg-red-500/20 text-red-400"
                }`}
              >
                {selectedPathData.difficulty === "beginner"
                  ? "Principiante"
                  : selectedPathData.difficulty === "intermediate"
                  ? "Intermedio"
                  : "Avanzado"}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex items-center gap-2 text-gray-400">
                <span>📍</span>
                <span>{selectedPathData.milestones.length} hitos</span>
              </div>
              <div className="flex items-center gap-2 text-gray-400">
                <span>⏱️</span>
                <span>{selectedPathData.estimatedHours} horas estimadas</span>
              </div>
              <div className="flex items-center gap-2 text-gray-400">
                <span>🎯</span>
                <span>{getPathProgress(selectedPathData)}% completado</span>
              </div>
              <div className="flex items-center gap-2 text-gray-400">
                <span>🏆</span>
                <span>
                  {
                    selectedPathData.milestones.filter((id) =>
                      isMilestoneCompleted(id)
                    ).length
                  }{" "}
                  / {selectedPathData.milestones.length} completados
                </span>
              </div>
            </div>

            {/* Progress bar general */}
            <div className="mt-4 w-full bg-slate-600 rounded-full h-3">
              <div
                className="bg-gradient-to-r from-cyan-500 to-blue-500 h-3 rounded-full transition-all duration-500"
                style={{ width: `${getPathProgress(selectedPathData)}%` }}
              />
            </div>
          </div>

          {/* Lista de Milestones en la Ruta */}
          <div className="space-y-3">
            <h4 className="text-xl font-bold text-white mb-4">
              📚 Hitos del Camino
            </h4>

            {selectedPathData.milestones.map((milestoneId, index) => {
              const milestone = AI_TIMELINE.find((m) => m.id === milestoneId);
              if (!milestone) return null;

              const isCompleted = isMilestoneCompleted(milestoneId);
              const isLocked =
                index > 0 &&
                !isMilestoneCompleted(selectedPathData.milestones[index - 1]);

              return (
                <div
                  key={milestoneId}
                  onClick={() =>
                    !isLocked &&
                    onSelectMilestone &&
                    onSelectMilestone(milestoneId)
                  }
                  className={`relative bg-slate-700/50 border rounded-lg p-4 transition-all ${
                    isCompleted
                      ? "border-green-500/50 bg-green-900/10"
                      : isLocked
                      ? "border-gray-600/30 opacity-50 cursor-not-allowed"
                      : "border-cyan-500/30 hover:border-cyan-400/60 cursor-pointer"
                  }`}
                >
                  <div className="flex items-start gap-4">
                    {/* Número de paso */}
                    <div
                      className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                        isCompleted
                          ? "bg-green-500 text-white"
                          : isLocked
                          ? "bg-gray-600 text-gray-400"
                          : "bg-cyan-500 text-white"
                      }`}
                    >
                      {isCompleted ? "✓" : index + 1}
                    </div>

                    {/* Información del milestone */}
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h5 className="font-bold text-white">
                          {milestone.title}
                        </h5>
                        <span className="text-xs text-gray-500">
                          ({milestone.year})
                        </span>
                        {isLocked && <span className="text-xs">🔒</span>}
                      </div>
                      <p className="text-sm text-gray-400">
                        {milestone.description}
                      </p>

                      {isCompleted && (
                        <div className="mt-2 text-xs text-green-400">
                          ✓ Completado
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Habilidades que se aprenden */}
          {selectedPathData.skills && selectedPathData.skills.length > 0 && (
            <div className="mt-6 bg-slate-700/30 rounded-lg p-4">
              <h4 className="text-lg font-bold text-white mb-3">
                🎯 Habilidades que Desarrollarás
              </h4>
              <div className="flex flex-wrap gap-2">
                {selectedPathData.skills.map((skill, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-cyan-500/20 text-cyan-300 rounded-full text-sm"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LearningPathVisualizer;
