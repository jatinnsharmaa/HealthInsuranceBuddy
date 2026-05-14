"use client";

import { useState } from "react";

interface FeedbackButtonsProps {
  onFeedback: (signal: "up" | "down", text?: string) => void;
}

export default function FeedbackButtons({ onFeedback }: FeedbackButtonsProps) {
  const [selected, setSelected] = useState<"up" | "down" | null>(null);
  const [showInput, setShowInput] = useState(false);
  const [text, setText] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSignal = (signal: "up" | "down") => {
    setSelected(signal);
    setShowInput(true);
  };

  const handleSubmit = () => {
    if (!selected) return;
    onFeedback(selected, text.trim() || undefined);
    setSubmitted(true);
    setShowInput(false);
  };

  if (submitted) {
    return <p className="text-xs text-slate-400 mt-2">Thanks for your feedback.</p>;
  }

  return (
    <div className="mt-3">
      <div className="flex gap-2">
        <button
          onClick={() => handleSignal("up")}
          className={`text-sm px-3 py-1 rounded-full border transition-colors ${
            selected === "up"
              ? "bg-emerald-50 border-emerald-300 text-emerald-700"
              : "border-slate-200 text-slate-500 hover:border-slate-300"
          }`}
        >
          👍 Helpful
        </button>
        <button
          onClick={() => handleSignal("down")}
          className={`text-sm px-3 py-1 rounded-full border transition-colors ${
            selected === "down"
              ? "bg-rose-50 border-rose-300 text-rose-700"
              : "border-slate-200 text-slate-500 hover:border-slate-300"
          }`}
        >
          👎 Not helpful
        </button>
      </div>

      {showInput && (
        <div className="mt-2 flex gap-2">
          <input
            type="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Optional: tell us more about this response."
            className="flex-1 text-sm border border-slate-200 rounded px-3 py-1.5 focus:outline-none focus:border-blue-400"
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
          <button
            onClick={handleSubmit}
            className="text-sm px-3 py-1.5 bg-slate-800 text-white rounded hover:bg-slate-700"
          >
            Submit
          </button>
        </div>
      )}
    </div>
  );
}
