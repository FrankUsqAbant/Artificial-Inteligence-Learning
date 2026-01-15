import React, { useState, useEffect, useCallback } from 'react';
import { AI_TIMELINE } from './data/ai_timeline';
import { Milestone } from './types';
import TimelineNode from './components/TimelineNode';
import LessonDrawer from './components/LessonDrawer';
import Dashboard from './components/Dashboard';
import NexusOracle from './components/NexusOracle';
import BootScreen from './components/BootScreen';
import LearningRoute from './components/LearningRoute';
import IdentityHUD from './components/IdentityHUD';
import { useNexusStore } from './hooks/useNexusStore';

const App: React.FC = () => {
  const [booting, setBooting] = useState(true);
  const { state, markAsMastered, progress } = useNexusStore();
  const [activeMilestone, setActiveMilestone] = useState<Milestone | null>(null);
  const [showDashboard, setShowDashboard] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const winScroll = window.scrollY || window.pageYOffset;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = height > 0 ? (winScroll / height) * 100 : 0;
      setScrollProgress(scrolled);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    
    const timer = setTimeout(() => {
      setBooting(false);
    }, 4000);

    return () => {
      window.removeEventListener('scroll', handleScroll);
      clearTimeout(timer);
    };
  }, []);

  const handleComplete = useCallback(() => {
    if (activeMilestone) {
      markAsMastered(activeMilestone.id);
      setActiveMilestone(null);
    }
  }, [activeMilestone, markAsMastered]);

  if (booting) return <BootScreen onComplete={() => setBooting(false)} />;

  return (
    <div className="min-h-screen selection:bg-green-500/30 bg-[#020202]">
      <header className="fixed top-0 left-0 right-0 z-[100] px-6 md:px-10 py-8 pointer-events-none">
        <div className="max-w-[1800px] mx-auto flex justify-between items-start">
          <IdentityHUD 
            onOpenDashboard={() => setShowDashboard(true)} 
            masteredCount={state.masteredIds.length} 
          />

          <div className="bg-black/80 backdrop-blur-3xl border border-white/10 p-5 rounded-3xl min-w-[200px] hidden lg:block pointer-events-auto shadow-2xl">
            <div className="flex justify-between items-end mb-2">
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest italic">Sync Status</span>
              <span className="text-lg font-black text-green-400 font-mono">{progress}%</span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div 
                className="h-full bg-green-500 transition-all duration-1000 shadow-[0_0_10px_#22c55e]" 
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        </div>
      </header>

      <LearningRoute masteredIds={state.masteredIds} />

      <main className="relative max-w-5xl mx-auto pt-[20vh] md:pt-[30vh] pb-[40vh] px-6">
        <div className="absolute left-1/2 -translate-x-1/2 top-0 bottom-0 w-[1px] bg-white/5"></div>
        <div 
          className="absolute left-1/2 -translate-x-1/2 top-0 w-[2px] bg-green-500 shadow-[0_0_15px_#22c55e] transition-all duration-300" 
          style={{ height: `${scrollProgress}%` }}
        ></div>

        <div className="space-y-[30vh] md:space-y-[40vh] relative z-10">
          {AI_TIMELINE.map((m, idx) => {
            const previousMilestoneId = idx > 0 ? AI_TIMELINE[idx - 1].id : null;
            const isAvailable = idx === 0 || (previousMilestoneId && state.masteredIds.includes(previousMilestoneId));
            
            return (
              <TimelineNode 
                key={m.id} 
                milestone={m} 
                isMastered={state.masteredIds.includes(m.id)}
                isAvailable={!!isAvailable}
                align={idx % 2 === 0 ? 'left' : 'right'}
                onClick={() => setActiveMilestone(m)}
              />
            );
          })}
        </div>
      </main>

      {activeMilestone && (
        <LessonDrawer 
          milestone={activeMilestone} 
          onClose={() => setActiveMilestone(null)}
          onComplete={handleComplete}
        />
      )}

      {showDashboard && (
        <Dashboard 
          masteredIds={state.masteredIds} 
          onClose={() => setShowDashboard(false)} 
        />
      )}
      
      <NexusOracle />
    </div>
  );
};

export default App;