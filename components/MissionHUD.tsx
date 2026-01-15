
import React from 'react';
import { Category } from '../types';
import { CATEGORY_THEMES } from '../data/aiElements';

interface MissionHUDProps {
  nextMission: {
    title: string;
    targetCategory: Category;
    current: number;
    required: number;
  } | null;
}

const MissionHUD: React.FC<MissionHUDProps> = ({ nextMission }) => {
  if (!nextMission) return null;

  const theme = CATEGORY_THEMES[nextMission.targetCategory];
  const progress = (nextMission.current / nextMission.required) * 100;

  return (
    <div className="fixed top-32 right-8 z-[40] animate-in slide-in-from-right duration-500">
      <div className="bg-slate-950/80 backdrop-blur-md border-l-4 border-cyan-500 p-4 w-64 shadow-2xl shadow-cyan-900/20 rounded-r-lg">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[9px] font-black text-cyan-500 uppercase tracking-[0.3em] animate-pulse">
            Mission_Objective
          </span>
          <i className="fa-solid fa-crosshairs text-[10px] text-cyan-500"></i>
        </div>
        
        <h4 className="text-white text-xs font-black uppercase italic tracking-tighter mb-3">
          {nextMission.title}
        </h4>

        <div className="space-y-2">
          <div className="flex justify-between items-end">
            <span className="text-[8px] font-mono text-slate-500 uppercase">Neural_Sync_Progress</span>
            <span className="text-[10px] font-mono text-white">{nextMission.current}/{nextMission.required}</span>
          </div>
          <div className="h-1 bg-slate-900 rounded-full overflow-hidden">
            <div 
              className="h-full bg-cyan-500 shadow-[0_0_8px_#06b6d4] transition-all duration-1000"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <p className="text-[8px] font-mono text-slate-600 leading-none pt-1">
            TARGET_NODE: <span className={theme.color}>{nextMission.targetCategory.toUpperCase()}</span>
          </p>
        </div>
      </div>
      
      {/* Decorative corner */}
      <div className="absolute -top-1 -right-1 w-2 h-2 border-t border-right border-cyan-500"></div>
    </div>
  );
};

export default MissionHUD;
