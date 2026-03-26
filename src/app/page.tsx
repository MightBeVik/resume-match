"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, FileSearch, Loader2 } from "lucide-react";

export default function InputPage() {
  const router = useRouter();
  const [resumeText, setResumeText] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = resumeText.length >= 50 && jobDescription.length >= 50 && !loading;

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_text: resumeText,
          job_description: jobDescription,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Analysis failed");
      }

      const data = await response.json();
      sessionStorage.setItem("matchResults", JSON.stringify(data));
      router.push("/results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header
        className="flex items-center gap-3 px-8 py-4 border-b bg-[var(--bg-card)]"
        style={{ borderColor: "var(--border)" }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "var(--text-primary)" }}
        >
          <FileSearch size={16} className="text-white" />
        </div>
        <span className="text-lg font-bold tracking-tight" style={{ color: "var(--text-primary)" }}>
          ResumeMatch
        </span>
        <span className="text-xs ml-1" style={{ color: "var(--text-muted)" }}>
          NLP-Powered Analysis
        </span>
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col items-center justify-center px-8 py-12">
        <div className="w-full max-w-4xl">
          {/* Title */}
          <div className="text-center mb-10">
            <h1 className="text-3xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
              Analyze Your Resume Match
            </h1>
            <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
              Paste your resume and a job description to see how well they align
            </p>
          </div>

          {/* Input cards */}
          <div className="grid grid-cols-2 gap-6">
            {/* Resume */}
            <div
              className="rounded-[var(--radius)] border overflow-hidden"
              style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
            >
              <div
                className="px-5 py-3 border-b"
                style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
              >
                <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  Your Resume
                </h2>
              </div>
              <div className="p-5">
                <textarea
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  className="w-full rounded-lg border p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[var(--accent)] transition-shadow"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                    background: "var(--bg-page)",
                  }}
                  rows={14}
                  placeholder="Paste your resume text here..."
                />
                <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
                  {resumeText.length} characters {resumeText.length < 50 && resumeText.length > 0 && "· minimum 50"}
                </p>
              </div>
            </div>

            {/* Job Description */}
            <div
              className="rounded-[var(--radius)] border overflow-hidden"
              style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
            >
              <div
                className="px-5 py-3 border-b"
                style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
              >
                <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  Job Description
                </h2>
              </div>
              <div className="p-5">
                <textarea
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  className="w-full rounded-lg border p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[var(--accent)] transition-shadow"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                    background: "var(--bg-page)",
                  }}
                  rows={14}
                  placeholder="Paste the job description here..."
                />
                <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
                  {jobDescription.length} characters {jobDescription.length < 50 && jobDescription.length > 0 && "· minimum 50"}
                </p>
              </div>
            </div>
          </div>

          {/* Error message */}
          {error && (
            <div
              className="mt-4 p-3 rounded-lg text-sm"
              style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}
            >
              {error}
            </div>
          )}

          {/* Analyze button */}
          <div className="mt-8 flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={!canSubmit}
              className="flex items-center gap-2 px-8 py-3 rounded-lg text-sm font-semibold text-white transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: "var(--text-primary)" }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Analyze Match
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
