
import React, { useEffect } from 'react';

interface MasteryNotificationProps {
  symbol: string;
  name: string;
  onClose: () => void;
}

const MasteryNotification: React.FC<MasteryNotificationProps> = ({ symbol, name, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-[200] animate-in slide-in-from-bottom-8 duration-500">
      <div className="bg-slate-950 border-2 border-cyan-500/50 p-1 rounded-xl shadow-[0_0_30px_rgba(6,182,212,0.3)]">
        <div className="bg-slate-900 px-8 py-4 rounded-lg flex items-center gap-6 border border-cyan-900/30">
          <div className="w-12 h-12 bg-cyan-500 rounded flex items-center justify-center shadow-[0_0_15px_rgba(6,182,212,0.5)]">
            <span className="text-black font-black text-xl italic">{symbol}</span>
          </div>
          <div>
            <h4 className="text-xs font-mono text-cyan-400 uppercase tracking-[0.3em] mb-1">Concept_Synapse_Success</h4>
            <p className="text-white font-black uppercase text-sm tracking-tighter">
              {name} ha sido grabado en el núcleo
            </p>
          </div>
          <div className="ml-4 pl-4 border-l border-slate-800">
            <i className="fa-solid fa-circle-check text-cyan-500 text-xl animate-pulse"></i>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MasteryNotification;
