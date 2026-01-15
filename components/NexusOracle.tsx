import React, { useState, useRef, useEffect } from "react";
import {
  answerQuestion,
  getSuggestedQuestions,
} from "../services/localKnowledgeService";

/**
 * NexusOracle - Sistema de Ayuda Local Inteligente
 * Sin dependencias de APIs externas
 */
const NexusOracle: React.FC = () => {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<
    { role: "user" | "oracle"; text: string; sources?: string[] }[]
  >([]);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history]);

  const suggestedQuestions = getSuggestedQuestions();

  const handleQuery = async (questionText?: string) => {
    const userText = questionText || query;
    if (!userText.trim() || loading) return;

    setQuery("");
    setShowSuggestions(false);
    setHistory((prev) => [...prev, { role: "user", text: userText }]);
    setLoading(true);

    // Simular delay para feedback visual
    setTimeout(() => {
      try {
        const answer = answerQuestion(userText);
        setHistory((prev) => [
          ...prev,
          {
            role: "oracle",
            text: answer.text,
            sources: answer.sources,
          },
        ]);
      } catch (error) {
        setHistory((prev) => [
          ...prev,
          {
            role: "oracle",
            text: "Error consultando la base de conocimiento local. Por favor, intenta reformular tu pregunta.",
          },
        ]);
      } finally {
        setLoading(false);
      }
    }, 300);
  };

  return (
    <div className="fixed bottom-8 right-8 z-[500] w-[420px] flex flex-col gap-4 pointer-events-none">
      {/* Ventana de chat */}
      <div
        className={`bg-black/95 backdrop-blur-3xl border border-white/10 rounded-3xl overflow-hidden flex flex-col transition-all duration-500 pointer-events-auto ${
          history.length > 0
            ? "h-[500px] opacity-100 translate-y-0"
            : "h-0 opacity-0 translate-y-10"
        }`}
      >
        <header className="px-6 py-4 border-b border-white/5 bg-white/[0.02] flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-[10px] font-bold text-white uppercase tracking-widest">
              Base de Conocimiento de IA
            </span>
          </div>
          <button
            onClick={() => setHistory([])}
            className="text-[10px] text-slate-500 hover:text-white uppercase font-bold transition-colors"
          >
            Limpiar
          </button>
        </header>

        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar"
        >
          {history.length === 0 && showSuggestions && (
            <div className="space-y-3">
              <p className="text-[11px] text-slate-500 uppercase tracking-wider font-bold">
                Preguntas Sugeridas:
              </p>
              {suggestedQuestions.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleQuery(q)}
                  className="w-full text-left p-3 rounded-xl bg-white/5 hover:bg-green-500/10 border border-white/10 hover:border-green-500/20 text-[11px] text-slate-400 hover:text-green-400 transition-all"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          {history.map((msg, i) => (
            <div
              key={i}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[85%] rounded-2xl text-[12px] leading-relaxed ${
                  msg.role === "user"
                    ? "bg-white/5 text-slate-300 p-4"
                    : "bg-green-500/10 border border-green-500/20 text-green-400 p-4 space-y-2"
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.text}</div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-green-500/20">
                    <p className="text-[9px] text-green-500/60 uppercase tracking-wider font-bold mb-1">
                      Fuentes:
                    </p>
                    {msg.sources.map((source, idx) => (
                      <div key={idx} className="text-[10px] text-green-500/80">
                        • {source}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-green-500/10 border border-green-500/20 rounded-2xl p-4">
                <div className="flex gap-1">
                  <div
                    className="w-2 h-2 rounded-full bg-green-500 animate-bounce"
                    style={{ animationDelay: "0ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 rounded-full bg-green-500 animate-bounce"
                    style={{ animationDelay: "150ms" }}
                  ></div>
                  <div
                    className="w-2 h-2 rounded-full bg-green-500 animate-bounce"
                    style={{ animationDelay: "300ms" }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input de pregunta */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleQuery();
        }}
        className="bg-black border border-white/10 p-2 rounded-full flex items-center gap-2 shadow-2xl pointer-events-auto group hover:border-green-500/30 transition-all"
      >
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Pregunta sobre la historia de la IA..."
          className="flex-1 bg-transparent border-none outline-none text-xs px-4 text-white placeholder:text-slate-600"
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="w-10 h-10 rounded-full bg-green-500 text-black flex items-center justify-center hover:bg-white transition-all disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <i className="fa-solid fa-paper-plane text-xs"></i>
        </button>
      </form>

      {/* Indicador de sistema local */}
      <div className="text-center pointer-events-none">
        <span className="text-[9px] text-slate-600 uppercase tracking-wider font-mono">
          🔒 Sistema Local • Sin APIs Externas
        </span>
      </div>
    </div>
  );
};

export default NexusOracle;
