
import React, { useState } from 'react';

const TensorLab: React.FC = () => {
  const [matrixSize, setMatrixSize] = useState(4);
  const [hoveredCell, setHoveredCell] = useState<{r: number, c: number} | null>(null);

  const generateMatrix = () => {
    return Array(matrixSize).fill(0).map(() => Array(matrixSize).fill(0).map(() => Math.random().toFixed(2)));
  };

  const matrix = generateMatrix();

  return (
    <div className="p-8 bg-slate-900/30 rounded-2xl border border-slate-800 space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-xs font-black text-purple-400 uppercase tracking-[0.3em]">Neural_Tensor_Lab_v1.0</h3>
        <div className="flex gap-2">
          {[3, 4, 5].map(size => (
            <button 
              key={size}
              onClick={() => setMatrixSize(size)}
              className={`w-8 h-8 rounded border text-[10px] font-mono ${matrixSize === size ? 'bg-purple-600 border-purple-400 text-white' : 'border-slate-800 text-slate-500 hover:border-slate-600'}`}
            >
              {size}x{size}
            </button>
          ))}
        </div>
      </div>

      <div className="relative group perspective-1000">
        <div className="grid gap-2 transform rotate-x-12 rotate-y--12 transition-transform duration-700 group-hover:rotate-x-0 group-hover:rotate-y-0"
             style={{ gridTemplateColumns: `repeat(${matrixSize}, 1fr)` }}>
          {matrix.map((row, r) => row.map((val, c) => (
            <div 
              key={`${r}-${c}`}
              onMouseEnter={() => setHoveredCell({r, c})}
              className={`aspect-square flex items-center justify-center text-[10px] font-mono border transition-all duration-300 rounded shadow-lg
                ${hoveredCell?.r === r || hoveredCell?.c === c ? 'bg-purple-900/40 border-purple-500 text-purple-200 scale-105 z-10' : 'bg-black/60 border-slate-800 text-slate-600 opacity-60'}
              `}
            >
              {val}
            </div>
          )))}
        </div>
        
        {/* Decorative Axes */}
        <div className="absolute -left-4 top-0 bottom-0 w-[1px] bg-gradient-to-t from-transparent via-purple-500 to-transparent"></div>
        <div className="absolute -bottom-4 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-purple-500 to-transparent"></div>
      </div>

      <div className="bg-black/50 p-4 rounded border border-purple-900/30">
        <p className="text-[10px] font-mono text-purple-300 leading-relaxed italic">
          <span className="text-white font-bold mr-2 uppercase">Kernel Log:</span> 
          Visualizando tensor de rango 2. Las celdas resaltadas representan el campo receptivo durante una operación de Producto Punto (Dot Product).
        </p>
      </div>
    </div>
  );
};

export default TensorLab;
