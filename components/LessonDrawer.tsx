import React, { useState, useEffect } from "react";
import { Milestone } from "../types";
import MarkdownRenderer from "./MarkdownRenderer";

interface Props {
  milestone: Milestone;
  onClose: () => void;
  onComplete: () => void;
}

const LessonDrawer: React.FC<Props> = ({ milestone, onClose, onComplete }) => {
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<{
    status: "success" | "fail" | null;
    text: string;
  }>({ status: null, text: "" });

  useEffect(() => {
    // Síntesis de voz local
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(
        `Explorando el hito de ${milestone.year}: ${milestone.title}`
      );
      utterance.lang = "es-ES";
      utterance.rate = 1.1;
      utterance.pitch = 0.9;
      window.speechSynthesis.speak(utterance);
    }
  }, [milestone.id]);

  const validateChallenge = async () => {
    if (!answer.trim() || loading) return;
    setLoading(true);

    // Simulación de procesamiento local
    setTimeout(() => {
      const userAnswer = answer.toLowerCase();
      const expected = milestone.practice.expected_logic.toLowerCase();

      // Validación simple: ¿La respuesta contiene la palabra clave esperada?
      if (userAnswer.includes(expected)) {
        setFeedback({
          status: "success",
          text: "¡Correcto! El registro histórico ha sido validado satisfactoriamente.",
        });
        setTimeout(onComplete, 1500);
      } else {
        setFeedback({
          status: "fail",
          text: `Análisis incompleto. Revisa la sección de 'Investigación Profunda' y enfócate en: ${milestone.practice.expected_logic}.`,
        });
      }
      setLoading(false);
    }, 800);
  };

  return (
    <div className="fixed inset-0 z-[600] flex justify-end">
      <div
        onClick={onClose}
        className="absolute inset-0 bg-black/95 backdrop-blur-xl"
      ></div>

      <div className="relative w-full max-w-2xl bg-[#030303] border-l border-white/5 h-full flex flex-col animate-in slide-in-from-right duration-500 overflow-hidden">
        <header className="p-10 border-b border-white/5 flex justify-between items-center bg-white/[0.01]">
          <div>
            <span className="text-[10px] font-mono text-green-500 uppercase tracking-widest block mb-2">
              {milestone.era} // {milestone.year}
            </span>
            <h2 className="text-3xl font-black text-white italic uppercase tracking-tighter leading-none">
              {milestone.title}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="w-12 h-12 rounded-full bg-white/5 flex items-center justify-center text-slate-500 hover:text-white transition-all"
          >
            <i className="fa-solid fa-xmark"></i>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto p-10 custom-scrollbar space-y-12">
          {milestone.deep_dive && (
            <section className="animate-in fade-in duration-700">
              <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.3em] mb-6 italic border-b border-white/5 pb-2">
                Investigación_Histórica
              </h4>
              <MarkdownRenderer content={milestone.deep_dive} />
            </section>
          )}

          <section className="p-8 bg-green-500/[0.02] rounded-3xl border border-green-500/10">
            <h4 className="text-[10px] font-black text-green-500/60 uppercase tracking-[0.3em] mb-4 italic">
              Archivo_Contextual
            </h4>
            <p className="text-lg text-slate-300 leading-relaxed italic border-l-2 border-green-500/40 pl-6">
              {milestone.historical_context}
            </p>
          </section>

          <section className="bg-black/80 p-8 rounded-3xl border border-white/5 shadow-2xl">
            <h4 className="text-[10px] font-black text-green-500 uppercase tracking-[0.3em] mb-4">
              Núcleo_Teórico
            </h4>
            <code className="text-2xl text-green-400 font-mono block text-center py-6 border-y border-white/5 my-4">
              {milestone.theory.math}
            </code>
            <p className="text-sm text-slate-400 mt-4 leading-relaxed">
              {milestone.theory.concept}
            </p>
          </section>

          <section className="space-y-6 pt-6 pb-20">
            <h4 className="text-[10px] font-black text-green-500 uppercase tracking-[0.3em]">
              Validación_de_Conocimiento
            </h4>
            <div className="bg-white/5 p-8 rounded-3xl border border-white/10 space-y-4">
              <p className="text-sm text-slate-300 font-medium italic">
                "{milestone.practice.challenge}"
              </p>
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Escribe tu respuesta basándote en los archivos superiores..."
                className="w-full bg-black/50 border border-white/10 rounded-2xl p-6 font-mono text-sm text-green-400 focus:border-green-500/30 outline-none h-32 resize-none"
              />
              <button
                onClick={validateChallenge}
                disabled={loading || !answer}
                className="w-full py-4 bg-green-600 text-black font-black uppercase tracking-[0.3em] text-[10px] rounded-xl hover:bg-white transition-all shadow-[0_0_20px_rgba(34,197,94,0.2)]"
              >
                {loading ? "VALIDANDO..." : "REGISTRAR APRENDIZAJE"}
              </button>
              {feedback.text && (
                <div
                  className={`p-5 rounded-xl border text-[11px] font-mono leading-relaxed animate-in zoom-in duration-300 ${
                    feedback.status === "success"
                      ? "bg-green-950/20 border-green-500/30 text-green-400"
                      : "bg-rose-950/20 border-rose-500/30 text-rose-400"
                  }`}
                >
                  {feedback.text}
                </div>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
};

export default LessonDrawer;
