"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, FileSearch, Loader2, Upload, FileText, X, ListFilter, MapPin } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Mode = "single" | "rank";

export default function InputPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("rank");
  const [resumeText, setResumeText] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [province, setProvince] = useState("ON");
  const [city, setCity] = useState("");
  const [jobTitleQuery, setJobTitleQuery] = useState("");
  const [limit, setLimit] = useState("20");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSubmit = resumeText.length >= 50 && !loading && !uploading && (mode === "rank" || jobDescription.length >= 50);

  const handleFileUpload = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported");
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/api/upload-resume`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
      }

      const data = await response.json();
      setResumeText(data.text);
      setUploadedFile(data.filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload file");
    } finally {
      setUploading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  }, [handleFileUpload]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const clearUpload = () => {
    setUploadedFile(null);
    setResumeText("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);

    try {
      const endpoint = mode === "single" ? `${API_BASE_URL}/api/analyze` : `${API_BASE_URL}/api/rank-jobs`;
      const payload = mode === "single"
        ? {
            resume_text: resumeText,
            job_description: jobDescription,
          }
        : {
            resume_text: resumeText,
            province: province || null,
            city: city || null,
            job_title_query: jobTitleQuery || null,
            limit: Number(limit) || 20,
            candidate_pool: Math.max(50, Number(limit) * 5 || 100),
          };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Analysis failed");
      }

      const data = await response.json();
      sessionStorage.setItem(
        "matchResults",
        JSON.stringify({
          mode,
          payload: data,
          filters: { province, city, jobTitleQuery, limit },
        }),
      );
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
        className="flex items-center justify-between px-8 py-4 border-b bg-[var(--bg-card)]"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "var(--text-primary)" }}
          >
            <FileSearch size={16} style={{ color: "var(--text-on-primary)" }} />
          </div>
          <span className="text-lg font-bold tracking-tight" style={{ color: "var(--text-primary)" }}>
            ResumeMatch
          </span>
          <span className="text-xs ml-1" style={{ color: "var(--text-muted)" }}>
            NLP-Powered Analysis
          </span>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => router.push("/about")}
            className="text-xs font-semibold px-4 py-1.5 rounded-lg transition-colors"
            style={{
              color: "var(--accent)",
              background: "var(--accent-light)",
              border: "1px solid var(--accent)",
            }}
          >
            About
          </button>
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-8 py-12">
        <div className="w-full max-w-4xl">
          {/* Title */}
          <div className="text-center mb-10">
            <h1 className="text-3xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
              Match Your Resume To Real Jobs
            </h1>
            <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
              Upload your resume, then either compare against one JD or rank it against the stored Canadian jobs corpus
            </p>
            <div className="mt-5 inline-flex rounded-xl border p-1" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
              {[
                { key: "rank", label: "Rank Stored Jobs", icon: ListFilter },
                { key: "single", label: "Single JD Analysis", icon: FileSearch },
              ].map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setMode(key as Mode)}
                  className="flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-colors"
                  style={{
                    background: mode === key ? "var(--text-primary)" : "transparent",
                    color: mode === key ? "var(--text-on-primary)" : "var(--text-secondary)",
                  }}
                >
                  <Icon size={15} />
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Input cards */}
          <div className={`grid gap-6 ${mode === "single" ? "grid-cols-2" : "grid-cols-[1.2fr_0.8fr]"}`}>
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
                {/* Upload zone */}
                {!uploadedFile && (
                  <>
                    <div
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onClick={() => fileInputRef.current?.click()}
                      className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-6 cursor-pointer transition-colors"
                      style={{
                        borderColor: dragOver ? "var(--accent)" : "var(--border)",
                        background: dragOver ? "var(--accent-light)" : "var(--bg-page)",
                      }}
                    >
                      {uploading ? (
                        <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
                      ) : (
                        <Upload size={24} style={{ color: "var(--text-muted)" }} />
                      )}
                      <p className="mt-2 text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                        {uploading ? "Extracting text..." : "Drop PDF here or click to upload"}
                      </p>
                      <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                        PDF files only
                      </p>
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) handleFileUpload(file);
                      }}
                    />

                    {/* Divider */}
                    <div className="flex items-center gap-3 my-4">
                      <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
                      <span className="text-xs uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>or paste text</span>
                      <div className="flex-1 h-px" style={{ background: "var(--border)" }} />
                    </div>
                  </>
                )}

                {/* Uploaded file indicator */}
                {uploadedFile && (
                  <div
                    className="flex items-center gap-3 p-3 rounded-lg mb-3"
                    style={{ background: "var(--success-bg)" }}
                  >
                    <FileText size={16} style={{ color: "var(--success-text)" }} />
                    <span className="text-xs font-medium flex-1 truncate" style={{ color: "var(--success-text)" }}>
                      {uploadedFile}
                    </span>
                    <button onClick={clearUpload} className="p-0.5 rounded hover:opacity-70">
                      <X size={14} style={{ color: "var(--success-text)" }} />
                    </button>
                  </div>
                )}

                <textarea
                  value={resumeText}
                  onChange={(e) => {
                    setResumeText(e.target.value);
                    if (uploadedFile) setUploadedFile(null);
                  }}
                  className="w-full rounded-lg border p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-[var(--accent)] transition-shadow"
                  style={{
                    borderColor: "var(--border)",
                    color: "var(--text-primary)",
                    background: "var(--bg-page)",
                  }}
                  rows={uploadedFile ? 8 : 6}
                  placeholder="Paste your resume text here..."
                />
                <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
                  {resumeText.length} characters {resumeText.length < 50 && resumeText.length > 0 && "· minimum 50"}
                </p>
              </div>
            </div>

            {/* Right panel */}
            <div
              className="rounded-[var(--radius)] border overflow-hidden"
              style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
            >
              <div
                className="px-5 py-3 border-b"
                style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
              >
                <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                  {mode === "single" ? "Job Description" : "Ranking Filters"}
                </h2>
              </div>
              <div className="p-5">
                {mode === "single" ? (
                  <>
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
                  </>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                        Province
                      </label>
                      <input
                        value={province}
                        onChange={(e) => setProvince(e.target.value.toUpperCase())}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                        placeholder="ON"
                      />
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                        City
                      </label>
                      <div className="relative">
                        <MapPin size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: "var(--text-muted)" }} />
                        <input
                          value={city}
                          onChange={(e) => setCity(e.target.value)}
                          className="w-full rounded-lg border py-2 pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                          style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                          placeholder="Toronto"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                        Job Title Contains
                      </label>
                      <input
                        value={jobTitleQuery}
                        onChange={(e) => setJobTitleQuery(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                        placeholder="frontend"
                      />
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                        Results Limit
                      </label>
                      <input
                        value={limit}
                        onChange={(e) => setLimit(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                        placeholder="20"
                      />
                    </div>
                    <div className="rounded-lg border p-3 text-sm" style={{ borderColor: "var(--border)", background: "var(--bg-page)", color: "var(--text-secondary)" }}>
                      This mode ranks your resume against the local `jobs.db` corpus built from Job Bank and OaSIS-enriched NOC data.
                    </div>
                  </div>
                )}
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
              className="flex items-center gap-2 px-8 py-3 rounded-lg text-sm font-semibold transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  {mode === "single" ? "Analyze Match" : "Rank Jobs"}
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
