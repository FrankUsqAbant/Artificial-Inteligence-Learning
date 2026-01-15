
import React, { useState } from 'react';

const STATIC_GALLERY = [
  { 
    id: 1, 
    title: "Estructura de Red", 
    url: "https://images.unsplash.com/photo-1677442136019-21780ecad995?q=80&w=800",
    desc: "Visualización de capas densas y propagación de señales."
  },
  { 
    id: 2, 
    title: "Abstracción Lógica", 
    url: "https://images.unsplash.com/photo-1620712943543-bcc4628c9757?q=80&w=800",
    desc: "Representación de la manipulación simbólica de la era de Dartmouth."
  },
  { 
    id: 3, 
    title: "Nodos Neuronales", 
    url: "https://images.unsplash.com/photo-1509228468518-180dd4864904?q=80&w=800",
    desc: "Concepto de pesos y bias en el Perceptrón."
  }
];

const ArchitectsCanvas: React.FC = () => {
  const [selected, setSelected] = useState(STATIC_GALLERY[0]);

  return (
    <div className="space-y-8 animate-in fade-in zoom-in duration-500">
      <div className="text-center space-y-2">
        <h3 className="text-3xl font-black text-white italic tracking-tighter uppercase">Galería_de_Conceptos</h3>
        <p className="text-[10px] font-mono text-green-500 uppercase tracking-[0.4em]">Visualización de Archivos Históricos</p>
      </div>

      <div className="max-w-5xl mx-auto flex flex-col lg:flex-row gap-8">
        <div className="flex-1 space-y-4">
          <div className="relative aspect-video bg-black border border-white/10 rounded-[2rem] overflow-hidden shadow-2xl">
            <img 
              key={selected.id}
              src={selected.url} 
              alt={selected.title} 
              className="w-full h-full object-cover animate-in fade-in duration-1000" 
            />
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-10">
              <h4 className="text-xl font-bold text-white uppercase">{selected.title}</h4>
              <p className="text-xs text-slate-400 mt-2 font-mono">{selected.desc}</p>
            </div>
          </div>
        </div>

        <div className="w-full lg:w-72 space-y-4">
          <h5 className="text-[10px] font-black text-slate-500 uppercase tracking-widest px-4">Seleccionar Archivo</h5>
          {STATIC_GALLERY.map((item) => (
            <button
              key={item.id}
              onClick={() => setSelected(item)}
              className={`w-full p-4 rounded-2xl border text-left transition-all ${
                selected.id === item.id 
                ? 'bg-green-500/10 border-green-500 text-green-400' 
                : 'bg-white/5 border-white/5 text-slate-500 hover:border-white/20'
              }`}
            >
              <p className="text-[10px] font-mono mb-1">ID: 00{item.id}</p>
              <p className="text-sm font-bold uppercase">{item.title}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ArchitectsCanvas;
