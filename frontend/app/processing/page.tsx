"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { getIngestStatus, IngestStatus } from "@/lib/api";

const STAGES = [
  { key: "parsing", label: "Parsing document" },
  { key: "chunking", label: "Extracting clauses" },
  { key: "indexing", label: "Indexing" },
  { key: "ready", label: "Ready" },
];

function getStageIndex(status: string): number {
  const map: Record<string, number> = { queued: 0, parsing: 0, chunking: 1, indexing: 2, ready: 3 };
  return map[status] ?? 0;
}

function ProcessingContent() {
  const router = useRouter();
  const params = useSearchParams();
  const jobId = params.get("job_id") || "";
  const policyId = params.get("policy_id") || "";
  const userUploaded = params.get("user_uploaded") === "true";

  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const interval = setInterval(async () => {
      try {
        const s = await getIngestStatus(jobId);
        setStatus(s);
        if (s.status === "ready") {
          clearInterval(interval);
          const pdfUrl = encodeURIComponent(`/pdfs/${policyId}.pdf`);
          router.push(`/chat?policy_id=${policyId}&pdf_url=${pdfUrl}&user_uploaded=${userUploaded}`);
        }
        if (s.status === "error") {
          clearInterval(interval);
          setError(s.error || "An error occurred during processing.");
        }
      } catch (e) {
        setError("Could not reach the server.");
        clearInterval(interval);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [jobId, policyId, router, userUploaded]);

  const currentStage = getStageIndex(status?.status || "queued");

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center px-4">
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-8 max-w-sm w-full text-center">
        <div className="text-4xl mb-4">⚙️</div>
        <h2 className="text-lg font-semibold text-slate-800 mb-6">Preparing your policy</h2>

        <div className="space-y-3 mb-6">
          {STAGES.map((stage, i) => {
            const done = i < currentStage;
            const active = i === currentStage && status?.status !== "error";
            return (
              <div key={stage.key} className="flex items-center gap-3">
                <div
                  className={`w-5 h-5 rounded-full flex items-center justify-center text-xs flex-shrink-0 ${
                    done
                      ? "bg-emerald-500 text-white"
                      : active
                      ? "bg-blue-500 text-white animate-pulse"
                      : "bg-slate-200 text-slate-400"
                  }`}
                >
                  {done ? "✓" : i + 1}
                </div>
                <span
                  className={`text-sm ${
                    done ? "text-emerald-700" : active ? "text-blue-700 font-medium" : "text-slate-400"
                  }`}
                >
                  {stage.label}
                </span>
              </div>
            );
          })}
        </div>

        {error && (
          <div className="text-sm text-rose-600 bg-rose-50 rounded-lg p-3">
            {error.includes("PDF") || error.includes("read")
              ? "We couldn't read this PDF. Try a text-based PDF or use the sample policy."
              : error}
          </div>
        )}

        {!error && (
          <p className="text-xs text-slate-400">
            This usually takes 30–60 seconds for a fresh upload.
          </p>
        )}
      </div>
    </main>
  );
}

export default function ProcessingPage() {
  return (
    <Suspense>
      <ProcessingContent />
    </Suspense>
  );
}
