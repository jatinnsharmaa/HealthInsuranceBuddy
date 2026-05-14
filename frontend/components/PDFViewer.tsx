"use client";

import { useState, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

interface PDFViewerProps {
  url: string;
  currentPage: number;
  onPageChange?: (page: number) => void;
}

export default function PDFViewer({ url, currentPage, onPageChange }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setLoading(false);
  }, []);

  const onDocumentLoadError = useCallback((err: Error) => {
    setError(err.message);
    setLoading(false);
  }, []);

  const goTo = (page: number) => {
    const clamped = Math.max(1, Math.min(page, numPages));
    onPageChange?.(clamped);
  };

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-slate-500 text-sm p-4 text-center">
        <p>Could not load PDF.<br />Make sure the policy has been uploaded and indexed.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-slate-100">
      {/* Page controls */}
      <div className="flex items-center justify-between px-4 py-2 bg-white border-b border-slate-200 shrink-0">
        <span className="text-sm text-slate-500 font-medium">Policy Document</span>
        {numPages > 0 && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => goTo(currentPage - 1)}
              disabled={currentPage <= 1}
              className="px-2 py-1 text-xs rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-40"
            >
              ‹ Prev
            </button>
            <span className="text-xs text-slate-600">
              Page {currentPage} of {numPages}
            </span>
            <button
              onClick={() => goTo(currentPage + 1)}
              disabled={currentPage >= numPages}
              className="px-2 py-1 text-xs rounded bg-slate-100 hover:bg-slate-200 disabled:opacity-40"
            >
              Next ›
            </button>
          </div>
        )}
      </div>

      {/* PDF render area */}
      <div className="flex-1 overflow-y-auto flex justify-center py-4">
        {loading && (
          <div className="flex items-center text-sm text-slate-400 mt-20">
            Loading document...
          </div>
        )}
        <Document
          file={url}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={onDocumentLoadError}
          loading=""
        >
          <Page
            pageNumber={currentPage}
            renderTextLayer={true}
            renderAnnotationLayer={true}
            width={480}
            className="shadow-lg rounded"
          />
        </Document>
      </div>
    </div>
  );
}
