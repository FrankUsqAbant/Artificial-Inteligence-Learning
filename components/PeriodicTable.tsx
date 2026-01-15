
import React, { useState, useMemo, useEffect } from 'react';
import { AI_ELEMENTS, CATEGORY_THEMES } from '../data/aiElements';
import { AIElement, UserRank, Category } from '../types';

interface PeriodicTableProps {
  onSelectElement: (element: AIElement) => void;
  masteredSymbols: string[];
  userRank: UserRank;
  onOpenGraph: () => void;
  onToggleMissions: () => void;
  unlockedCategories: Category[];
}

const PeriodicTable: React.FC<PeriodicTableProps> = ({ 
  onSelectElement, 
  masteredSymbols, 
  userRank, 
  onOpenGraph, 
  onToggleMissions,
  unlockedCategories
}) => {
  const [search, setSearch] = useState('');
  const [hoveredElement, setHoveredElement] = useState<AIElement | null>(null);

  const activeCategory = hoveredElement?.category || (unlockedCategories.length > 0 ? unlockedCategories[unlockedCategories.length - 1] : 'math-foundation');
  const categoryTheme = CATEGORY_THEMES[activeCategory];

  const filteredElements = useMemo(() => {
    const term = search.toLowerCase();
    return AI_ELEMENTS.filter(el => 
      el.name.toLowerCase().includes(term) || 
      el.symbol.toLowerCase().includes(term)
    );
  }, [search]);

  const masteryPercentage = Math.round((masteredSymbols.length / AI_ELEMENTS.length) * 100);

  return (
    <div className="p-4 md:p-8 min-h-screen bg-[#050505] flex flex-col relative overflow-hidden">
      {/* Context Top Bar */}
      <div className={`fixed top-0 left-0 right-0 z-50 p-4 border-b ${categoryTheme.border} bg-black/60 backdrop-blur-md transition-all duration-500`}>
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-6">
            <span className={`text-[10px] font-mono uppercase tracking-[0.5em] px-3 py-1 bg-slate-900 border ${categoryTheme.border} ${categoryTheme.color}`}>
              Learning_Phase: {activeCategory.replace('-', '_')}
            </span>
            <p className="text-[11px] text-slate-400 font-mono hidden md:block">
              <span className="text-white font-bold mr-2">CONTEXT:</span> {categoryTheme.desc}
            </p>
          </div>
          <div className="flex items-center gap-3">
             <i className="fa-solid fa-circle-nodes text-[10px] animate-pulse text-cyan-500"></i>
             <span className="text-[8px] font-mono text-slate-600 uppercase tracking-widest">Neural_Sync_Active</span>
          </div>
        </div>
      </div>

      <div className="max-w-[1400px] mx-auto w-full flex-1 relative z-10 mt-20">
        <header className="mb-12 flex flex-col lg:flex-row justify-between items-end gap-8">
          <div className="space-y-4 w-full lg:w-auto">
            <div className="space-y-1">
              <h1 className="text-6xl font-black tracking-tighter text-white uppercase leading-none italic">
                NEXUS<span className="text-cyan-500">.AI</span>
              </h1>
              <p className="text-slate-500 font-mono text-[10px] uppercase tracking-[0.5em]">Neural Learning Interface</p>
            </div>
            
            <div className="flex gap-3">
              <div className="relative group flex-1 max-w-md">
                <i className="fa-solid fa-terminal absolute left-4 top-1/2 -translate-y-1/2 text-slate-600 group-focus-within:text-cyan-500 transition-colors"></i>
                <input 
                  type="text" 
                  placeholder="QUERY_KNOWLEDGE..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full bg-slate-900/30 border border-slate-800 rounded py-3 pl-12 pr-4 text-xs font-mono text-white focus:border-cyan-500/50 transition-all uppercase"
                />
              </div>
              <button onClick={onOpenGraph} className="px-4 bg-slate-900 border border-slate-800 rounded text-cyan-500 hover:bg-cyan-500/10 transition-all">
                <i className="fa-solid fa-diagram-project"></i>
              </button>
            </div>
          </div>
          
          <div className="flex gap-8 p-6 bg-slate-900/20 rounded-xl border border-slate-800/50">
            <div className="text-center">
              <p className="text-slate-500 text-[9px] font-black uppercase tracking-widest mb-2">Sync_Status</p>
              <div className="flex items-center gap-3">
                <p className="text-white font-mono text-3xl font-bold">{masteryPercentage}%</p>
                <div className="w-24 h-1 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500" style={{ width: `${masteryPercentage}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        </header>

        <div className="overflow-x-auto pb-8 custom-scrollbar">
          <div 
            className="grid gap-2 mx-auto" 
            style={{ 
              gridTemplateColumns: 'repeat(18, minmax(75px, 1fr))',
              gridTemplateRows: 'repeat(7, 95px)',
              width: 'max-content'
            }}
          >
            {filteredElements.map((el) => {
              const theme = CATEGORY_THEMES[el.category];
              const isMastered = masteredSymbols.includes(el.symbol);
              const isLocked = !unlockedCategories.includes(el.category);
              
              return (
                <button
                  key={el.symbol}
                  onClick={() => !isLocked && onSelectElement(el)}
                  onMouseEnter={() => setHoveredElement(el)}
                  onMouseLeave={() => setHoveredElement(null)}
                  disabled={isLocked}
                  style={{ gridColumn: el.position.x, gridRow: el.position.y }}
                  className={`relative group transition-all duration-300 border rounded flex flex-col items-center justify-center
                    ${isLocked 
                      ? 'bg-black border-slate-900 opacity-20 cursor-not-allowed grayscale' 
                      : isMastered 
                        ? `bg-slate-900 ${theme.border} ${theme.glow} hover:scale-110 z-10` 
                        : 'bg-[#0a0a0a] border-slate-800 hover:border-slate-500 hover:scale-105 opacity-60 hover:opacity-100 z-0'
                    }
                  `}
                >
                  <span className={`absolute top-1 left-1 text-[8px] font-mono ${isMastered ? theme.color : 'text-slate-700'}`}>{el.atomic_number}</span>
                  <span className={`text-2xl font-black italic tracking-tighter ${isMastered ? theme.color : 'text-slate-500'}`}>{el.symbol}</span>
                  <span className={`text-[8px] font-bold uppercase truncate w-full px-1 text-center ${isMastered ? 'text-slate-300' : 'text-slate-700'}`}>{isLocked ? 'LOCKED' : el.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PeriodicTable;
