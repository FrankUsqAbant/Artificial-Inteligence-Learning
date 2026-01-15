
import React from 'react';

interface GlassProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  hover?: boolean;
}

export const GlassCard: React.FC<GlassProps> = ({ children, className = "", onClick, hover = true }) => (
  <div 
    onClick={onClick}
    className={`
      relative overflow-hidden backdrop-blur-md border border-white/5 bg-white/[0.02]
      ${hover ? 'hover:bg-white/[0.04] hover:border-white/20 transition-all duration-500' : ''}
      ${className}
    `}
  >
    {children}
  </div>
);

export const Badge: React.FC<{ children: React.ReactNode; color?: string }> = ({ children, color = "cyan" }) => (
  <span className={`text-[9px] font-mono px-3 py-1 rounded-full uppercase tracking-widest border border-${color}-500/20 bg-${color}-500/10 text-${color}-400`}>
    {children}
  </span>
);
