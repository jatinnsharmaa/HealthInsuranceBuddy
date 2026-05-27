"use client";

import { useState } from "react";
import { ChatMessage, Citation } from "@/lib/api";
import FeedbackButtons from "./FeedbackButtons";

interface MessageBubbleProps {
  message: ChatMessage;
  onCitationClick: (page: number) => void;
  onFeedback: (signal: "up" | "down", text?: string) => void;
  isLast: boolean;
}

function VerdictBadge({ text }: { text: string }) {
  const lower = text.toLowerCase();
  let verdict = "";
  let color = "";

  if (/^\*\*verdict:\s*yes\b/i.test(text) || /^verdict:\s*yes\b/i.test(text)) {
    verdict = "✓ Yes"; color = "bg-emerald-100 text-emerald-800 border-emerald-200";
  } else if (/^\*\*verdict:\s*no\b/i.test(text) || /^verdict:\s*no\b/i.test(text)) {
    verdict = "✗ No"; color = "bg-rose-100 text-rose-800 border-rose-200";
  } else if (/^\*\*verdict:\s*partial\b/i.test(text) || /^verdict:\s*partial\b/i.test(text)) {
    verdict = "◑ Partial"; color = "bg-amber-100 text-amber-800 border-amber-200";
  }

  if (!verdict) return null;

  return (
    <span className={`inline-block text-xs font-semibold px-2.5 py-0.5 rounded-full border mb-2 ${color}`}>
      {verdict}
    </span>
  );
}

/** Split LLM output into {reasoning, answer}.
 *  Everything before the first "Verdict:" line is reasoning.
 *  If no "Verdict:" is found, the whole content is the answer.
 *  Used by ThinkingBlock and FollowUpSuggestions components (Tasks 3–5).
 */
function splitContent(content: string): { reasoning: string; answer: string } {
  // Match "Verdict:" at start-of-string OR after a newline.
  // If no Verdict found, treat entire content as the answer (no reasoning to separate).
  const match = content.match(/(^|\n)(\*\*Verdict:|Verdict:)/im);
  if (!match || match.index === undefined) {
    return { reasoning: "", answer: content };
  }
  // If matched at position 0 (Verdict at very start), there's no reasoning block
  if (match.index === 0) {
    return { reasoning: "", answer: content.trim() };
  }
  return {
    reasoning: content.slice(0, match.index).trim(),
    answer: content.slice(match.index + 1).trim(),
  };
}

/** Return a human-readable label for the most recent Step N: STEPNAME found in content.
 *  Used by ThinkingBlock component (Tasks 3–5).
 */
function detectCurrentStep(content: string): string {
  const stepMap: Record<string, string> = {
    THINK: "Thinking…",
    RETRIEVE: "Retrieving clauses…",
    EVALUATE_POLICY: "Evaluating policy…",
    GENERATE: "Generating answer…",
  };
  // Find all "Step N: STEPNAME" occurrences, take the last one
  const matches = [...content.matchAll(/Step\s+\d+:\s+([A-Z_]+)/gi)];
  if (matches.length === 0) return "Working…";
  const lastStep = matches[matches.length - 1][1].toUpperCase();
  return stepMap[lastStep] ?? "Working…";
}

function renderContent(content: string, onCitationClick: (page: number) => void) {
  // Remove the verdict line since we show it as a badge
  const withoutVerdict = content
    .replace(/^\*\*Verdict:[^*]*\*\*\n?/im, "")
    .replace(/^Verdict:[^\n]*\n?/im, "");

  // Split on citation refs like [Clause 3.2, Page 11]
  const parts = withoutVerdict.split(/(\[(?:Clause|Section|Annexure)[^\]]+Page\s+\d+\])/gi);

  return parts.map((part, i) => {
    const match = part.match(/\[(?:Clause|Section|Annexure)\s*([\d.IVXivx]+),?\s*Page\s*(\d+)\]/i);
    if (match) {
      const page = parseInt(match[2]);
      return (
        <button
          key={i}
          onClick={() => onCitationClick(page)}
          className="inline-flex items-center gap-1 text-xs bg-blue-50 text-blue-700 border border-blue-200 rounded px-1.5 py-0.5 mx-0.5 hover:bg-blue-100 transition-colors"
          title={`Jump to page ${page}`}
        >
          📄 {part.slice(1, -1)}
        </button>
      );
    }
    // Format **bold** text, then convert \n to <br />
    return (
      <span key={i}>
        {part.split(/\*\*([^*]+)\*\*/g).map((s, j) => {
          if (j % 2 === 1) return <strong key={j}>{s}</strong>;
          // Preserve newlines as <br />
          return s.split("\n").map((line, k, arr) => (
            <span key={k}>
              {line}
              {k < arr.length - 1 && <br />}
            </span>
          ));
        })}
      </span>
    );
  });
}

/** Shown while the agent is streaming — pulses the current step name.
 * Uses detectCurrentStep() defined above.
 */
function ThinkingIndicator({ content }: { content: string }) {
  const label = detectCurrentStep(content);
  return (
    <div className="flex items-center gap-1.5 text-xs text-slate-400 mb-2">
      <span className="animate-spin inline-block">⚙</span>
      <span>{label}</span>
      <span className="animate-pulse">▌</span>
    </div>
  );
}

/** Shown after streaming completes — collapsible reasoning trace.
 * Uses useState to toggle open/closed. Used by MessageBubble (Task 5).
 */
function ThinkingDisclosure({ reasoning }: { reasoning: string }) {
  const [open, setOpen] = useState(false);

  if (!reasoning) return null;

  // Bold step labels like "Step 1: THINK"
  const formatted = reasoning.split(/^(Step\s+\d+:\s+\S+)/gm).map((part, i) =>
    /^Step\s+\d+:/i.test(part)
      ? <strong key={i} className="text-slate-500">{part}</strong>
      : <span key={i}>{part}</span>
  );

  return (
    <div className="mb-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600 transition-colors"
      >
        <span>💭 View reasoning</span>
        <span className={`transition-transform ${open ? "rotate-90" : ""}`}>›</span>
      </button>
      {open && (
        <pre className="mt-1 font-mono text-xs text-slate-400 bg-slate-50 rounded p-3 whitespace-pre-wrap overflow-x-auto">
          {formatted}
        </pre>
      )}
    </div>
  );
}

const FOLLOW_UP_MAP: Record<string, string[]> = {
  waiting_period: [
    "Does portability reduce my waiting period?",
    "What counts as a pre-existing disease?",
  ],
  maternity: [
    "Is there a waiting period for maternity cover?",
    "Are newborn expenses covered from day one?",
  ],
  exclusion: [
    "Are there permanent exclusions that can never be covered?",
    "What's the room rent sub-limit?",
  ],
  hospital_network: [
    "How do I find a network hospital near me?",
    "What happens if I go to a non-network hospital?",
  ],
  claim: [
    "What documents are needed for a reimbursement claim?",
    "What's the deadline for filing a claim?",
  ],
  room_rent: [
    "How does the room rent sub-limit affect my claim?",
    "Is ICU treatment covered without a sub-limit?",
  ],
  premium: [
    "Will my premium increase after a claim?",
    "How does portability work?",
  ],
  fallback: [
    "What are the main exclusions in this policy?",
    "Is there a room rent sub-limit?",
    "Can I add a family member mid-term?",
  ],
};

const TOPIC_KEYWORDS: Array<{ topic: string; keywords: string[] }> = [
  { topic: "waiting_period", keywords: ["waiting period", "ped", "pre-existing", "36 month", "pre existing"] },
  { topic: "maternity", keywords: ["maternity", "pregnancy", "newborn", "delivery"] },
  { topic: "exclusion", keywords: ["exclusion", "not covered", "excluded", "permanent"] },
  { topic: "hospital_network", keywords: ["cashless", "network hospital", "tpa", "network provider"] },
  { topic: "claim", keywords: ["claim", "reimbursement", "discharge", "document"] },
  { topic: "room_rent", keywords: ["room rent", "sub-limit", "icu", "ward"] },
  { topic: "premium", keywords: ["premium", "renewal", "portability", "no-claim bonus"] },
];

/** Detect topic from the user question + agent answer for follow-up suggestion mapping. */
function detectTopic(question: string, answer: string): string {
  const combined = (question + " " + answer).toLowerCase();
  for (const { topic, keywords } of TOPIC_KEYWORDS) {
    if (keywords.some((kw) => combined.includes(kw))) return topic;
  }
  return "fallback";
}

interface FollowUpSuggestionsProps {
  question: string;
  answer: string;
  onSend: (text: string) => void;
}

/** Shown below the last agent message — contextual follow-up question pills.
 * Used by MessageBubble (Task 5).
 */
function FollowUpSuggestions({ question, answer, onSend }: FollowUpSuggestionsProps) {
  const topic = detectTopic(question, answer);
  const suggestions = FOLLOW_UP_MAP[topic] ?? FOLLOW_UP_MAP.fallback;

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {suggestions.map((q) => (
        <button
          key={q}
          onClick={() => onSend(q)}
          className="text-xs border border-slate-200 rounded-full px-3 py-1 text-slate-600 hover:border-blue-300 hover:text-blue-700 hover:bg-blue-50 transition-colors"
        >
          → {q}
        </button>
      ))}
    </div>
  );
}

export default function MessageBubble({ message, onCitationClick, onFeedback, isLast }: MessageBubbleProps) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end mb-4">
        <div className="bg-slate-800 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[75%] text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  const isLiveData = message.webFetchStatus === "succeeded";
  const webFailed = message.webFetchStatus === "failed";

  return (
    <div className="flex justify-start mb-4">
      <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%] shadow-sm">
        {/* Verdict badge */}
        <VerdictBadge text={message.content} />

        {/* Answer */}
        <div className="text-sm text-slate-800 leading-relaxed whitespace-pre-wrap">
          {message.isStreaming ? (
            <span>{message.content}<span className="animate-pulse">▌</span></span>
          ) : (
            renderContent(message.content, onCitationClick)
          )}
        </div>

        {/* Live data tag */}
        {isLiveData && (
          <div className="mt-2 text-xs text-teal-700 bg-teal-50 border border-teal-200 rounded px-2 py-1 inline-flex items-center gap-1">
            🌐 Includes live data from insurer/IRDAI website
          </div>
        )}

        {/* Web fetch failure warning */}
        {webFailed && (
          <div className="mt-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
            ⚠️ Could not reach the insurer&apos;s website for live data. Answer is based on the policy document only.
          </div>
        )}

        {/* Disclaimer */}
        {!message.isStreaming && (
          <p className="mt-3 text-xs text-slate-400 italic">
            This response is generated by AI and may contain mistakes. Please double-check before acting on it.
          </p>
        )}

        {/* Feedback */}
        {!message.isStreaming && isLast && (
          <FeedbackButtons onFeedback={onFeedback} />
        )}
      </div>
    </div>
  );
}
