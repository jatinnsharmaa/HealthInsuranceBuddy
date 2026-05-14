"use client";

import { useState, useEffect, useCallback, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { v4 as uuidv4 } from "uuid";
import dynamic from "next/dynamic";
import ChatPane from "@/components/ChatPane";
import { ChatMessage, streamChat, submitFeedback, getPDFUrl } from "@/lib/api";

const PDFViewer = dynamic(() => import("@/components/PDFViewer"), { ssr: false });

function ChatContent() {
  const router = useRouter();
  const params = useSearchParams();
  const policyId = params.get("policy_id") || "";
  const pdfUrlParam = params.get("pdf_url") || "";
  const userUploaded = params.get("user_uploaded") === "true";

  const [sessionId] = useState(() => uuidv4());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [pdfUrl, setPdfUrl] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [lastRetrievalLog, setLastRetrievalLog] = useState<unknown[]>([]);
  const [lastWebFetchStatus, setLastWebFetchStatus] = useState("not_triggered");

  useEffect(() => {
    if (pdfUrlParam) {
      const url = decodeURIComponent(pdfUrlParam);
      // If it's a relative path, prefix with API base
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      setPdfUrl(url.startsWith("http") ? url : `${apiBase}${url}`);
    } else if (policyId) {
      setPdfUrl(getPDFUrl(policyId));
    }
  }, [pdfUrlParam, policyId]);

  const handleSend = useCallback(
    async (text: string) => {
      if (!policyId || busy) return;
      setBusy(true);

      const userMsg: ChatMessage = { role: "user", content: text };
      setMessages((prev) => [...prev, userMsg]);

      const agentMsg: ChatMessage = {
        role: "agent",
        content: "",
        isStreaming: true,
        citations: [],
      };
      setMessages((prev) => [...prev, agentMsg]);

      try {
        let fullContent = "";
        for await (const event of streamChat(
          sessionId,
          policyId,
          text,
          messages,
        )) {
          if (event.type === "chunk" && event.content) {
            fullContent += event.content;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                content: fullContent,
              };
              return updated;
            });
          } else if (event.type === "done") {
            const retrieval = event.retrieval_log || [];
            const webStatus = event.web_fetch_status || "not_triggered";
            setLastRetrievalLog(retrieval as unknown[]);
            setLastWebFetchStatus(webStatus);
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                content: fullContent,
                isStreaming: false,
                citations: event.citations || [],
                webFetchStatus: webStatus,
              };
              return updated;
            });
          }
        }
      } catch (e) {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "agent",
            content: "Sorry, something went wrong. Please try again.",
            isStreaming: false,
          };
          return updated;
        });
      } finally {
        setBusy(false);
      }
    },
    [policyId, sessionId, messages, busy]
  );

  const handleFeedback = useCallback(
    async (msgIndex: number, signal: "up" | "down", text?: string) => {
      const conversation = messages.slice(0, msgIndex + 1).map((m, i) => ({
        turn: Math.floor(i / 2) + 1,
        role: m.role,
        content: m.content,
      }));

      await submitFeedback({
        session_id: sessionId,
        policy_id: policyId,
        conversation,
        retrieval_log: lastRetrievalLog,
        web_fetch_status: lastWebFetchStatus,
        feedback_signal: signal,
        feedback_text: text,
      });
    },
    [sessionId, policyId, messages, lastRetrievalLog, lastWebFetchStatus]
  );

  return (
    <div className="flex flex-col h-screen bg-slate-100">
      {/* Top bar */}
      <header className="flex items-center justify-between px-4 py-2 bg-white border-b border-slate-200 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-800">Health Insurance Interpreter</span>
          <span className="text-xs text-slate-400">{policyId}</span>
        </div>
        <button
          onClick={() => router.push("/")}
          className="text-xs text-slate-500 hover:text-slate-700 px-3 py-1.5 rounded-lg hover:bg-slate-100 transition-colors"
        >
          Start over
        </button>
      </header>

      {/* Two-pane layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: chat */}
        <div className="w-1/2 flex flex-col border-r border-slate-200">
          <ChatPane
            messages={messages}
            onSend={handleSend}
            onCitationClick={setCurrentPage}
            onFeedback={handleFeedback}
            disabled={busy}
            userUploaded={userUploaded}
          />
        </div>

        {/* Right: PDF viewer */}
        <div className="w-1/2 flex flex-col">
          {pdfUrl ? (
            <PDFViewer
              url={pdfUrl}
              currentPage={currentPage}
              onPageChange={setCurrentPage}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-slate-400 text-sm">
              No PDF loaded
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense>
      <ChatContent />
    </Suspense>
  );
}
