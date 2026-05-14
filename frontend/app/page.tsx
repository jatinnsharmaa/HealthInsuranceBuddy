"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { uploadPDF, getSamplePolicy } from "@/lib/api";

export default function LandingPage() {
  const router = useRouter();
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    if (!file.name.endsWith(".pdf")) {
      setError("Please upload a PDF file.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const job = await uploadPDF(file);
      router.push(`/processing?job_id=${job.job_id}&policy_id=${job.policy_id}&user_uploaded=true`);
    } catch {
      setError("Upload failed. Please try again.");
      setLoading(false);
    }
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const handleSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const sample = await getSamplePolicy();
      router.push(`/chat?policy_id=${sample.policy_id}&pdf_url=${encodeURIComponent(sample.pdf_url)}`);
    } catch {
      setError("Sample policy not available. Please upload your own.");
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex flex-col items-center justify-center px-4">
      <div className="max-w-xl w-full">
        <div className="text-center mb-10">
          <div className="inline-block bg-blue-600 text-white text-xs font-semibold px-3 py-1 rounded-full mb-4">
            AI-Powered
          </div>
          <h1 className="text-4xl font-bold text-slate-900 leading-tight mb-3">
            Understand your health insurance policy in plain English.
          </h1>
          <p className="text-slate-500 text-base">
            Ask questions about coverage, waiting periods, exclusions, and claims — and get answers with exact clause references.
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 mb-4">
          <label
            className={`flex flex-col items-center justify-center border-2 border-dashed rounded-xl p-8 cursor-pointer transition-colors ${
              dragging ? "border-blue-400 bg-blue-50" : "border-slate-200 hover:border-blue-300"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
            />
            <div className="text-3xl mb-3">📄</div>
            <p className="text-sm font-medium text-slate-700">
              {loading ? "Uploading..." : "Upload your policy PDF"}
            </p>
            <p className="text-xs text-slate-400 mt-1">Drag and drop or click to browse</p>
          </label>

          <div className="flex items-center gap-3 my-4">
            <div className="flex-1 h-px bg-slate-200" />
            <span className="text-xs text-slate-400">or</span>
            <div className="flex-1 h-px bg-slate-200" />
          </div>

          <button
            onClick={handleSample}
            disabled={loading}
            className="w-full py-3 rounded-xl border border-slate-200 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            Try sample policy (Care Insurance Supreme)
          </button>

          {error && <p className="text-xs text-rose-600 mt-3 text-center">{error}</p>}
        </div>

        <p className="text-center text-xs text-slate-400">
          Your document is processed in memory and deleted after the session. We don&apos;t store it.
        </p>
      </div>
    </main>
  );
}
