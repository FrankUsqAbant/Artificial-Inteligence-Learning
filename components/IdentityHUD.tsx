
import React from 'react';

interface Props {
  onOpenDashboard: () => void;
  masteredCount: number;
}

const IdentityHUD: React.FC<Props> = ({ onOpenDashboard, masteredCount }) => (
  <div onClick={onOpenDashboard} className="relative group cursor-pointer pointer-events-auto transition-transform hover:scale-105 active:scale-95">
    <div className="flex items-center gap-4 bg-black/40 p-2 pr-6 rounded-full border border-white/5 backdrop-blur-xl hover:border-green-500/30 transition-colors">
      <div className="w-12 h-12 relative">
        <div className="absolute inset-0 bg-green-500/20 rounded-full animate-pulse"></div>
        <div className="absolute inset-[2px] bg-black rounded-full flex items-center justify-center border border-green-500/30">
          <div className="text-green-500 font-black text-sm italic">FA</div>
        </div>
      </div>
      <div>
        <p className="text-[10px] font-bold text-white uppercase tracking-widest">Frank Abanto</p>
        <p className="text-[8px] text-slate-500 uppercase font-mono">{masteredCount} Sincronizados</p>
      </div>
    </div>
  </div>
);

export default IdentityHUD;
