
import React from 'react';

interface Props {
  content: string;
}

const MarkdownRenderer: React.FC<Props> = ({ content }) => {
  // Procesamiento simple de markdown (negritas, listas, saltos de línea)
  const formatContent = (text: string) => {
    return text
      .split('\n')
      .map((line, i) => {
        // Listas
        if (line.trim().startsWith('- ')) {
          return <li key={i} className="ml-4 mb-2 text-slate-400">{line.replace('- ', '')}</li>;
        }
        // Encabezados
        if (line.trim().startsWith('### ')) {
          return <h3 key={i} className="text-lg font-black text-green-400 mt-6 mb-3 uppercase tracking-tighter italic">{line.replace('### ', '')}</h3>;
        }
        // Negritas
        const parts = line.split(/(\*\*.*?\*\*)/g);
        return (
          <p key={i} className="mb-4 text-slate-300 leading-relaxed">
            {parts.map((part, j) => 
              part.startsWith('**') && part.endsWith('**') 
                ? <strong key={j} className="text-white font-bold">{part.slice(2, -2)}</strong>
                : part
            )}
          </p>
        );
      });
  };

  return <div className="prose-nexus">{formatContent(content)}</div>;
};

export default MarkdownRenderer;
