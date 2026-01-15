import React, { useState } from "react";
import { AI_TIMELINE } from "../data/ai_timeline";
import ArchitectsCanvas from "./ArchitectsCanvas";
import PeriodicTable from "./PeriodicTable";
import LearningPathVisualizer from "./LearningPathVisualizer";
import { AIElement } from "../types";

interface DashboardProps {
  onClose: () => void;
  masteredIds: string[];
  onSelectMilestone?: (milestoneId: string) => void;
}

const Dashboard: React.FC<DashboardProps> = ({
  onClose,
  masteredIds,
  onSelectMilestone,
}) => {
  const [activeTab, setActiveTab] = useState<
    "INDICE" | "RUTAS" | "VISUALIZADOR" | "SISTEMAS"
  >("INDICE");

  return (
    <div className="fixed inset-0 z-[300] flex items-center justify-center p-6 md:p-12 overflow-hidden">
      <div
        onClick={onClose}
        className="absolute inset-0 bg-black/95 backdrop-blur-xl"
      ></div>

      <div className="relative w-full max-w-7xl bg-[#030303] border border-white/10 rounded-[2.5rem] overflow-hidden flex flex-col h-[85vh] shadow-2xl">
        <header className="px-10 py-6 border-b border-white/5 flex flex-wrap justify-between items-center gap-4">
          <h2 className="text-xl font-black text-white uppercase italic tracking-tighter">
            Panel de Control de IA
          </h2>

          <div className="flex bg-white/5 p-1 rounded-2xl">
            {(["INDICE", "RUTAS", "VISUALIZADOR", "SISTEMAS"] as const).map(
              (tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-2 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all ${
                    activeTab === tab
                      ? "bg-green-500 text-black shadow-[0_0_15px_#22c55e]"
                      : "text-slate-500 hover:text-white"
                  }`}
                >
                  {tab}
                </button>
              )
            )}
          </div>

          <button
            onClick={onClose}
            className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-slate-500 hover:text-white transition-colors"
          >
            <i className="fa-solid fa-xmark"></i>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-10 custom-scrollbar">
          {activeTab === "INDICE" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {AI_TIMELINE.map((m) => (
                <div
                  key={m.id}
                  className={`p-6 rounded-3xl border transition-all ${
                    masteredIds.includes(m.id)
                      ? "bg-green-500/5 border-green-500/20"
                      : "bg-white/[0.02] border-white/5 opacity-50"
                  }`}
                >
                  <div className="flex justify-between items-start mb-4">
                    <span className="text-[10px] font-mono text-slate-500">
                      {m.year}
                    </span>
                    {masteredIds.includes(m.id) && (
                      <i className="fa-solid fa-circle-check text-green-500 text-[10px]"></i>
                    )}
                  </div>
                  <h4 className="text-sm font-bold text-white uppercase mb-2 tracking-tight">
                    {m.title}
                  </h4>
                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    {m.description}
                  </p>
                </div>
              ))}
            </div>
          )}

          {activeTab === "RUTAS" && (
            <LearningPathVisualizer
              onSelectMilestone={(id) => {
                if (onSelectMilestone) {
                  onSelectMilestone(id);
                  onClose(); // Cerrar dashboard al seleccionar milestone
                }
              }}
            />
          )}

          {activeTab === "VISUALIZADOR" && <ArchitectsCanvas />}

          {activeTab === "SISTEMAS" && (
            <div className="h-full">
              <PeriodicTable
                onSelectElement={(el: AIElement) => console.log(el)}
                masteredSymbols={[]}
                userRank="architect"
                unlockedCategories={[
                  "math-foundation",
                  "neural-architectures",
                  "data-engineering",
                ]}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
