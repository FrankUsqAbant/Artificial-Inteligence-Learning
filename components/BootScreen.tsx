
import React, { useState, useEffect } from 'react';

const BootScreen: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(onComplete, 500);
          return 100;
        }
        return prev + 5;
      });
    }, 50);
    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <div className="fixed inset-0 bg-black z-[999] flex flex-col items-center justify-center p-8 overflow-hidden">
      <div className="max-w-xs w-full space-y-8 text-center">
        <div className="w-16 h-16 bg-green-600 rounded-2xl flex items-center justify-center mx-auto shadow-[0_0_30px_#16a34a]">
          <span className="text-black font-black text-2xl italic">FA</span>
        </div>
        <div className="space-y-2">
          <h1 className="text-white font-black text-2xl tracking-tighter uppercase">Frank Abanto</h1>
          <p className="text-[10px] text-slate-500 uppercase tracking-widest">Cargando cronología histórica...</p>
        </div>
        <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
          <div className="h-full bg-green-500 transition-all duration-300" style={{ width: `${progress}%` }}></div>
        </div>
      </div>
    </div>
  );
};

export default BootScreen;
