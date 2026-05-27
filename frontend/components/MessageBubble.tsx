"use client";

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
 */
function splitContent(content: string): { reasoning: string; answer: string } {
  const match = content.match(/\n(\*\*Verdict:|Verdict:)/i);
  if (!match || match.index === undefined) {
    return { reasoning: "", answer: content };
  }
  return {
    reasoning: content.slice(0, match.index).trim(),
    answer: content.slice(match.index + 1).trim(), // +1 to skip the leading \n
  };
}

/** Return a human-readable label for the most recent Step N: STEPNAME found in content. */
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
