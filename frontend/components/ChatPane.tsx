"use client";

import { useRef, useEffect } from "react";
import { ChatMessage } from "@/lib/api";
import MessageBubble from "./MessageBubble";

const STARTER_QUESTIONS = [
  "What's my waiting period for pre-existing conditions?",
  "Is maternity covered under this policy?",
  "Can I get cashless treatment at a specific hospital?",
  "What's not covered by this policy?",
];

interface ChatPaneProps {
  messages: ChatMessage[];
  onSend: (text: string) => void;
  onCitationClick: (page: number) => void;
  onFeedback: (msgIndex: number, signal: "up" | "down", text?: string) => void;
  disabled?: boolean;
  userUploaded?: boolean;
}

export default function ChatPane({
  messages,
  onSend,
  onCitationClick,
  onFeedback,
  disabled,
  userUploaded,
}: ChatPaneProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const val = inputRef.current?.value.trim();
    if (val) {
      onSend(val);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const agentMessages = messages.filter((m) => m.role === "agent");
  const lastAgentIdx = messages.length - 1 - [...messages].reverse().findIndex((m) => m.role === "agent");

  return (
    <div className="flex flex-col h-full bg-slate-50">
      {/* Unverified banner for user-uploaded docs */}
      {userUploaded && (
        <div className="bg-amber-50 border-b border-amber-200 text-amber-800 text-xs px-4 py-2 text-center">
          ⚠️ This policy hasn&apos;t been evaluated for accuracy. Answers are unverified.
        </div>
      )}

      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-4">
            <p className="text-slate-500 text-sm font-medium">Ask me anything about your policy.</p>
            <div className="grid grid-cols-1 gap-2 w-full max-w-sm">
              {STARTER_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => onSend(q)}
                  className="text-left text-sm px-4 py-2.5 rounded-xl border border-slate-200 bg-white hover:border-blue-300 hover:bg-blue-50 transition-colors text-slate-700"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            message={msg}
            previousUserMessage={
              msg.role === "agent" && i > 0 && messages[i - 1].role === "user"
                ? messages[i - 1].content
                : undefined
            }
            onCitationClick={onCitationClick}
            onFeedback={(signal, text) => onFeedback(i, signal, text)}
            onSend={onSend}
            isLast={i === lastAgentIdx && msg.role === "agent"}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <form onSubmit={handleSubmit} className="px-4 py-3 bg-white border-t border-slate-200 shrink-0">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            placeholder="Ask a question about your policy..."
            disabled={disabled}
            className="flex-1 text-sm border border-slate-200 rounded-xl px-4 py-2.5 focus:outline-none focus:border-blue-400 disabled:bg-slate-50 disabled:text-slate-400"
          />
          <button
            type="submit"
            disabled={disabled}
            className="px-4 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
