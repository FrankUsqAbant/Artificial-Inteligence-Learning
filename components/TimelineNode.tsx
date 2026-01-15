
import React from 'react';
import { Milestone } from '../types';
import { GlassCard, Badge } from './Shared';

interface Props {
  milestone: Milestone;
  isMastered: boolean;
  isAvailable: boolean;
  align: 'left' | 'right';
  onClick: () => void;
}

const TimelineNode: React.FC<Props> = React.memo(({ milestone, isMastered, isAvailable, align, onClick }) => {
  const isCrisis = milestone.type === 'crisis';

  return (
    <div className={`relative flex items-center ${align === 'left' ? 'flex-row' : 'flex-row-reverse'} w-full group`}>
      <div className={`w-[45%] transition-all duration-1000 ${isAvailable ? 'opacity-100 translate-y-0' : 'opacity-10 translate-y-20'}`}>
        <GlassCard 
          onClick={onClick}
          className={`p-10 rounded-[2.5rem] cursor-pointer shadow-2xl
            ${isCrisis ? 'border-rose-900/20 shadow-[0_0_50px_rgba(244,63,94,0.05)]' : ''}
            ${isMastered ? 'border-emerald-500/30' : ''}
            ${!isAvailable ? 'pointer-events-none saturate-0' : ''}
          `}
        >
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
            <div className="scan-line"></div>
          </div>

          <div className="flex justify-between items-start mb-8 relative z-10">
            <span className="text-6xl font-black text-white/5 italic tracking-tighter leading-none group-hover:text-white/10 transition-all">
              {milestone.year}
            </span>
            <Badge color={isCrisis ? 'rose' : 'white'}>{milestone.era}</Badge>
          </div>

          <h3 className={`text-3xl font-black uppercase tracking-tighter mb-4 relative z-10 leading-none transition-colors
            ${isCrisis ? 'text-rose-200' : isMastered ? 'text-emerald-400' : 'text-white'}`}>
            {milestone.title}
          </h3>
          
          <p className="text-[13px] text-slate-500 leading-relaxed font-medium mb-10 relative z-10 line-clamp-2">
            {milestone.description}
          </p>
          
          <div className="flex items-center justify-between border-t border-white/5 pt-8 relative z-10">
             <div className="flex items-center gap-3">
                <div className={`w-2 h-2 rounded-full ${isMastered ? 'bg-emerald-500 shadow-[0_0_10px_#10b981]' : isAvailable ? 'bg-cyan-500 animate-pulse shadow-[0_0_10px_#06b6d4]' : 'bg-slate-800'}`}></div>
                <span className="text-[10px] font-mono text-slate-600 uppercase tracking-widest font-bold">
                  {isMastered ? 'SYNC_COMPLETE' : isAvailable ? 'READY' : 'LOCKED'}
                </span>
             </div>
             <div className={`flex items-center gap-2 text-[10px] font-black uppercase tracking-widest transition-all group-hover:gap-4 ${isCrisis ? 'text-rose-500' : 'text-cyan-500'}`}>
                <span>EXPLORAR</span>
                <i className="fa-solid fa-arrow-right-long"></i>
             </div>
          </div>
        </GlassCard>
      </div>

      <div className="absolute left-1/2 -translate-x-1/2 flex items-center justify-center z-20">
        <div 
          className={`w-20 h-20 rounded-full border-2 bg-black flex items-center justify-center transition-all duration-700
          ${isMastered ? 'border-emerald-500 shadow-[0_0_40px_rgba(16,185,129,0.4)]' : 
            isAvailable ? 'border-cyan-500 shadow-[0_0_40px_rgba(6,182,212,0.4)]' : 
            'border-white/5 opacity-10'}
          ${isCrisis ? 'border-rose-500 shadow-[0_0_40px_rgba(244,63,94,0.4)]' : ''}`}
        >
          {isMastered ? <i className="fa-solid fa-check text-emerald-500 text-2xl"></i> :
           isCrisis ? <i className="fa-solid fa-bolt-lightning text-rose-500 text-2xl animate-pulse"></i> :
           !isAvailable ? <i className="fa-solid fa-lock text-slate-900 text-lg"></i> :
           <div className="w-10 h-10 border border-cyan-500/30 rounded-lg flex items-center justify-center animate-[spin_10s_linear_infinite]">
             <div className="w-2 h-2 bg-cyan-500 rounded-full"></div>
           </div>}
        </div>
      </div>
    </div>
  );
});

export default TimelineNode;
