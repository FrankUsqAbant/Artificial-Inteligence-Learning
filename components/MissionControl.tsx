
import React from 'react';
import { NEXUS_MISSIONS } from '../data/missions';
import { Category } from '../types';
import { CATEGORY_THEMES, AI_ELEMENTS } from '../data/aiElements';

interface MissionControlProps {
  masteredElements: string[];
  unlockedCategories: Category[];
  onClose: () => void;
}

const MissionControl: React.FC<MissionControlProps> = ({ masteredElements, unlockedCategories, onClose }) => {
  // Calculamos cuántos elementos de cada categoría se han masterizado
  const getCategoryMastery = (cat: Category) => {
    return AI_ELEMENTS.filter(el => el.category === cat && masteredElements.includes(el.symbol)).length;
  };

  const totalMastered = masteredElements.length;

  return (
    <div className="fixed inset-y-0 right-0 w-85 bg-slate-950 border-l border-slate-800 z-[100] p-6 shadow-2xl animate-in slide-in-from-right duration-300 flex flex-col">
      <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
        <div>
          <h3 className="text-sm font-black text-white uppercase tracking-widest italic">Mission_Control</h3>
          <p className="text-[9px] font-mono text-cyan-500">NEURAL_SYNC_ADVISOR</p>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-white transition-colors">
          <i className="fa-solid fa-xmark"></i>
        </button>
      </header>

      <div className="flex-1 space-y-6 overflow-y-auto pr-2 custom-scrollbar">
        {NEXUS_MISSIONS.map((mission) => {
          const isUnlocked = unlockedCategories.includes(mission.unlocksCategory);
          const theme = CATEGORY_THEMES[mission.unlocksCategory];
          const req = mission.requiredMasteryCount;
          const progress = Math.min(100, (totalMastered / (req || 1)) * 100);
          
          return (
            <div 
              key={mission.id}
              className={`p-4 rounded-xl border transition-all duration-300 relative overflow-hidden group ${
                isUnlocked 
                  ? 'bg-slate-900/40 border-slate-800' 
                  : 'bg-black/40 border-slate-900/50 opacity-60'
              }`}
            >
              {!isUnlocked && (
                <div className="absolute top-2 right-2 flex items-center gap-1 bg-black/60 px-2 py-0.5 rounded border border-slate-800">
                  <i className="fa-solid fa-lock text-[8px] text-slate-500"></i>
                  <span className="text-[8px] font-mono text-slate-500">{totalMastered}/{req} REQ</span>
                </div>
              )}

              <div className="mb-2">
                <span className={`text-[9px] font-mono uppercase tracking-tighter ${isUnlocked ? theme.color : 'text-slate-700'}`}>
                  {isUnlocked ? 'SYNC_ACTIVE' : 'SYNC_LOCKED'}
                </span>
              </div>

              <h4 className={`text-xs font-black uppercase mb-1 ${isUnlocked ? 'text-white' : 'text-slate-600'}`}>
                {mission.title}
              </h4>
              <p className="text-[10px] text-slate-500 leading-tight mb-4 font-mono group-hover:text-slate-400 transition-colors">
                {mission.description}
              </p>

              <div className="space-y-2">
                <div className="flex justify-between text-[8px] font-mono">
                  <span className="text-slate-600">INTEL_CATEGORY</span>
                  <span className={isUnlocked ? theme.color : 'text-slate-800'}>{mission.unlocksCategory.replace('-', '_').toUpperCase()}</span>
                </div>
                <div className="h-1 bg-slate-800/50 rounded-full overflow-hidden">
                  <div 
                    className={`h-full transition-all duration-1000 ${isUnlocked ? theme.color.replace('text', 'bg') : 'bg-slate-900'}`}
                    style={{ width: `${isUnlocked ? 100 : progress}%` }}
                  ></div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <footer className="mt-8 pt-6 border-t border-slate-900">
        <div className="p-4 bg-cyan-950/10 border border-cyan-900/20 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <p className="text-[10px] font-black text-cyan-500 uppercase tracking-widest">Global_Nexus_Sync</p>
            <span className="text-[10px] font-mono text-white">{totalMastered}/{AI_ELEMENTS.length}</span>
          </div>
          <div className="h-2 bg-slate-900 rounded-full overflow-hidden p-[2px]">
            <div 
              className="h-full bg-cyan-500 rounded-full shadow-[0_0_10px_rgba(6,182,212,0.5)] transition-all duration-1000"
              style={{ width: `${(totalMastered / AI_ELEMENTS.length) * 100}%` }}
            ></div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default MissionControl;
