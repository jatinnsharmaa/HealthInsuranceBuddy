const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface IngestJob {
  job_id: string;
  policy_id: string;
}

export interface IngestStatus {
  status: "queued" | "parsing" | "chunking" | "indexing" | "ready" | "error";
  progress: number;
  policy_id?: string;
  page_count?: number;
  chunk_count?: number;
  deferred_urls?: Array<{ url: string; governs: string; clause: string; page_number: number }>;
  error?: string;
}

export interface SamplePolicy {
  policy_id: string;
  display_name: string;
  pdf_url: string;
}

export interface Citation {
  clause: string;
  page: number;
}

export interface ChatMessage {
  role: "user" | "agent";
  content: string;
  citations?: Citation[];
  webFetchStatus?: string;
  isStreaming?: boolean;
}

export async function uploadPDF(file: File, policyId?: string): Promise<IngestJob> {
  const form = new FormData();
  form.append("file", file);
  if (policyId) form.append("policy_id", policyId);

  const res = await fetch(`${API_URL}/api/ingest`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getIngestStatus(jobId: string): Promise<IngestStatus> {
  const res = await fetch(`${API_URL}/api/ingest/status/${jobId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSamplePolicy(): Promise<SamplePolicy> {
  const res = await fetch(`${API_URL}/api/ingest/sample`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function* streamChat(
  sessionId: string,
  policyId: string,
  message: string,
  conversationHistory: ChatMessage[],
  retrievalMode = "hybrid_rerank"
): AsyncGenerator<{ type: string; content?: string; citations?: Citation[]; retrieval_log?: unknown[]; web_fetch_status?: string }> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      policy_id: policyId,
      message,
      conversation_history: conversationHistory.map((m, i) => ({
        turn: Math.floor(i / 2) + 1,
        role: m.role,
        content: m.content,
      })),
      retrieval_mode: retrievalMode,
    }),
  });

  if (!res.ok) throw new Error(await res.text());
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          yield JSON.parse(line.slice(6));
        } catch {
          // skip malformed SSE
        }
      }
    }
  }
}

export async function submitFeedback(data: {
  session_id: string;
  policy_id: string;
  conversation: ChatMessage[];
  retrieval_log: unknown[];
  web_fetch_status: string;
  feedback_signal: "up" | "down";
  feedback_text?: string;
}): Promise<void> {
  await fetch(`${API_URL}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export function getPDFUrl(policyId: string): string {
  return `${API_URL}/pdfs/${policyId}.pdf`;
}
