
import React, { useState, useMemo } from 'react';
import { AI_ELEMENTS, CATEGORY_THEMES } from '../data/aiElements';
import { AIElement } from '../types';

interface DependencyGraphProps {
  onSelectElement: (element: AIElement) => void;
  masteredSymbols: string[];
  onClose: () => void;
}

const DependencyGraph: React.FC<DependencyGraphProps> = ({ onSelectElement, masteredSymbols, onClose }) => {
  const [hoveredSymbol, setHoveredSymbol] = useState<string | null>(null);

  // Dimensiones del lienzo
  const width = 1200;
  const height = 800;

  // Calculamos posiciones de los nodos para el grafo (distribución circular por categorías)
  const nodes = useMemo(() => {
    const categories = Array.from(new Set(AI_ELEMENTS.map(e => e.category)));
    return AI_ELEMENTS.map((el, i) => {
      const catIndex = categories.indexOf(el.category);
      const angle = (catIndex * (2 * Math.PI / categories.length)) + (i * 0.1);
      const radius = 150 + (catIndex * 80);
      return {
        ...el,
        gx: width / 2 + Math.cos(angle) * radius,
        gy: height / 2 + Math.sin(angle) * radius,
      };
    });
  }, []);

  const connections = useMemo(() => {
    const lines: { x1: number, y1: number, x2: number, y2: number, color: string, active: boolean, id: string }[] = [];
    nodes.forEach(node => {
      node.prerequisites.forEach(pre => {
        const target = nodes.find(n => n.symbol === pre.symbol);
        if (target) {
          const isActive = hoveredSymbol === node.symbol || hoveredSymbol === target.symbol;
          lines.push({
            x1: node.gx, y1: node.gy,
            x2: target.gx, y2: target.gy,
            color: CATEGORY_THEMES[node.category].color.replace('text', 'stroke'),
            active: isActive,
            id: `${node.symbol}-${target.symbol}`
          });
        }
      });
    });
    return lines;
  }, [nodes, hoveredSymbol]);

  return (
    <div className="fixed inset-0 bg-[#050505] z-50 flex flex-col">
      <header className="p-6 border-b border-slate-800 flex justify-between items-center bg-black/50 backdrop-blur-md">
        <div>
          <h2 className="text-2xl font-black text-white italic tracking-tighter uppercase">Neural_Dependency_Map</h2>
          <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Visualizing concept architecture</p>
        </div>
        <button onClick={onClose} className="px-6 py-2 bg-slate-900 border border-slate-800 rounded-full text-xs font-mono text-slate-400 hover:text-cyan-500 transition-colors">
          [EXIT_GRAPH_VIEW]
        </button>
      </header>

      <div className="flex-1 relative overflow-hidden cursor-grab active:cursor-grabbing">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
          {/* Conexiones */}
          {connections.map(line => (
            <line
              key={line.id}
              x1={line.x1} y1={line.y1}
              x2={line.x2} y2={line.y2}
              className={`transition-all duration-500 ${line.active ? 'stroke-cyan-500 stroke-[2px] opacity-100' : 'stroke-slate-800 stroke-[1px] opacity-30'}`}
              style={{ stroke: line.active ? undefined : 'currentColor' }}
            />
          ))}

          {/* Nodos */}
          {nodes.map(node => {
            const theme = CATEGORY_THEMES[node.category];
            const isMastered = masteredSymbols.includes(node.symbol);
            const isHighlighted = hoveredSymbol === node.symbol;

            return (
              <g 
                key={node.symbol} 
                className="cursor-pointer"
                onMouseEnter={() => setHoveredSymbol(node.symbol)}
                onMouseLeave={() => setHoveredSymbol(null)}
                onClick={() => onSelectElement(node)}
              >
                <circle
                  cx={node.gx} cy={node.gy}
                  r={isHighlighted ? 25 : 18}
                  className={`transition-all duration-300 ${isMastered ? theme.color.replace('text', 'fill') + ' opacity-80' : 'fill-slate-900'}`}
                  stroke={isHighlighted ? '#06b6d4' : '#1e293b'}
                  strokeWidth={isHighlighted ? 3 : 1}
                />
                <text
                  x={node.gx} y={node.gy + 4}
                  textAnchor="middle"
                  className={`text-[10px] font-black pointer-events-none transition-colors ${isMastered ? 'fill-black' : 'fill-slate-500'}`}
                >
                  {node.symbol}
                </text>
                {isHighlighted && (
                  <text
                    x={node.gx} y={node.gy + 40}
                    textAnchor="middle"
                    className="text-[10px] font-mono fill-cyan-500 uppercase font-bold"
                  >
                    {node.name}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* Legend Panel */}
        <div className="absolute bottom-8 left-8 p-6 bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-2xl max-w-xs space-y-4">
          <h4 className="text-xs font-black text-white uppercase tracking-widest border-b border-slate-800 pb-2">Navegación Táctica</h4>
          <p className="text-[10px] text-slate-400 leading-relaxed font-mono">
            Los nodos representan unidades de conocimiento. Las líneas indican el flujo de pre-requisitos necesarios para la "Ingeniería de Sistemas de IA".
          </p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-cyan-500"></div>
              <span className="text-[9px] text-slate-300 font-bold uppercase">Concepto Activo</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-slate-700"></div>
              <span className="text-[9px] text-slate-500 font-bold uppercase">Concepto Latente</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DependencyGraph;
