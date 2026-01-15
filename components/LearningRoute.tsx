import React from 'react';
import { AI_TIMELINE } from '../data/ai_timeline';

interface Props {
  masteredIds: string[];
}

const LearningRoute: React.FC<Props> = ({ masteredIds }) => {
  const eras = Array.from(new Set(AI_TIMELINE.map(m => m.era)));

  return (
    <div className="fixed top-32 left-8 z-[40] hidden xl:block">
      <div className="bg-black/80 backdrop-blur-xl border border-white/5 p-6 rounded-3xl w-64 shadow-2xl overflow-y-auto max-h-[70vh] custom-scrollbar">
        <h3 className="text-[10px] font-black text-green-500 uppercase tracking-[0.3em] mb-6 border-b border-white/5 pb-2 italic">Mapa Estelar de IA</h3>
        
        <div className="space-y-8">
          {eras.map(era => (
            <div key={era} className="space-y-3">
              <h4 className="text-[8px] font-black text-slate-500 uppercase tracking-widest">{era}</h4>
              <div className="space-y-2">
                {AI_TIMELINE.filter(m => m.era === era).map((m) => {
                  const isMastered = masteredIds.includes(m.id);
                  const currentIndex = AI_TIMELINE.findIndex(item => item.id === m.id);
                  const previousMilestoneId = currentIndex > 0 ? AI_TIMELINE[currentIndex - 1].id : null;
                  const isAvailable = currentIndex === 0 || (previousMilestoneId && masteredIds.includes(previousMilestoneId));
                  
                  return (
                    <div key={m.id} className="flex items-center gap-3 group cursor-help">
                      <div className={`w-1.5 h-1.5 rounded-full transition-all duration-500 ${
                        isMastered ? 'bg-green-500 shadow-[0_0_8px_#22c55e]' : isAvailable ? 'bg-cyan-500/50' : 'bg-white/5'
                      }`}></div>
                      <div className="flex flex-col min-w-0">
                        <span className={`text-[10px] font-bold truncate ${isMastered ? 'text-white' : isAvailable ? 'text-slate-400' : 'text-slate-600'}`}>
                          {m.title}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LearningRoute;