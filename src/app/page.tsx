"use client";

import { useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, FileSearch, Loader2, Upload, FileText, X, ListFilter, MapPin, Users, Search } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Mode = "single" | "rank" | "bulk";

interface BulkResumeEntry {
  name: string;
  resume_text: string;
}

export default function InputPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("rank");

  // Single / rank mode state
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

  // Bulk mode state
  const [bulkResumes, setBulkResumes] = useState<BulkResumeEntry[]>([]);
  const [bulkUploading, setBulkUploading] = useState(false);
  const [bulkDragOver, setBulkDragOver] = useState(false);
  const [bulkProvince, setBulkProvince] = useState("");
  const [bulkCity, setBulkCity] = useState("");
  const [bulkKeyword, setBulkKeyword] = useState("");
  const [bulkCategory, setBulkCategory] = useState("");
  const [bulkMaxJobs, setBulkMaxJobs] = useState("30");
  const bulkFileInputRef = useRef<HTMLInputElement>(null);

  const canSubmit = resumeText.length >= 50 && !loading && !uploading && (mode === "rank" || jobDescription.length >= 50);
  const canBulkSubmit = bulkResumes.length > 0 && !loading && !bulkUploading;

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

  // Bulk upload handler
  const handleBulkFileUpload = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported");
      return;
    }
    setBulkResumes((prev) => {
      if (prev.length >= 20) { setError("Maximum 20 resumes"); return prev; }
      if (prev.some((r) => r.name === file.name)) { setError(`"${file.name}" already added`); return prev; }
      return prev; // real upload happens async below
    });
    setBulkUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API_BASE_URL}/api/upload-resume`, { method: "POST", body: form });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Upload failed"); }
      const data = await res.json();
      setBulkResumes((prev) => {
        if (prev.length >= 20) return prev;
        if (prev.some((r) => r.name === data.filename)) return prev;
        return [...prev, { name: data.filename, resume_text: data.text }];
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setBulkUploading(false);
    }
  }, []);

  const handleBulkSubmit = async () => {
    if (bulkResumes.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const payload = {
        resumes: bulkResumes,
        province: bulkProvince || null,
        city: bulkCity || null,
        keyword: bulkKeyword || null,
        broad_category: bulkCategory || null,
        max_jobs: Number(bulkMaxJobs) || 30,
      };
      const res = await fetch(`${API_BASE_URL}/api/bulk-match`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) { const err = await res.json(); throw new Error(err.detail || "Bulk match failed"); }
      const data = await res.json();
      sessionStorage.setItem("bulkMatchResult", JSON.stringify(data));
      sessionStorage.removeItem("selectedJobIds");
      sessionStorage.removeItem("selectedJobCards");
      router.push("/leaderboard");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setLoading(false);
    }
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
          <button
            onClick={() => router.push("/jobs")}
            className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
            style={{ color: "var(--text-secondary)", background: "var(--bg-hover)" }}
          >
            <Search size={13} />
            Job Search
          </button>
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
                { key: "bulk", label: "Bulk Match", icon: Users },
              ].map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => { setMode(key as Mode); setError(null); }}
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

          {/* ── BULK MODE ── */}
          {mode === "bulk" && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-5">
                {/* Multi-PDF upload */}
                <div className="rounded-[var(--radius)] border overflow-hidden" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                  <div className="px-5 py-3 border-b" style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}>
                    <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Resumes ({bulkResumes.length}/20)</h2>
                  </div>
                  <div className="p-5">
                    <div
                      onDrop={(e) => { e.preventDefault(); setBulkDragOver(false); Array.from(e.dataTransfer.files).forEach(handleBulkFileUpload); }}
                      onDragOver={(e) => { e.preventDefault(); setBulkDragOver(true); }}
                      onDragLeave={() => setBulkDragOver(false)}
                      onClick={() => bulkFileInputRef.current?.click()}
                      className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-5 cursor-pointer transition-colors"
                      style={{ borderColor: bulkDragOver ? "var(--accent)" : "var(--border)", background: bulkDragOver ? "var(--accent-light)" : "var(--bg-page)" }}
                    >
                      {bulkUploading ? <Loader2 size={22} className="animate-spin" style={{ color: "var(--accent)" }} /> : <Upload size={22} style={{ color: "var(--text-muted)" }} />}
                      <p className="mt-2 text-sm font-medium" style={{ color: "var(--text-secondary)" }}>
                        {bulkUploading ? "Extracting text…" : "Drop PDFs or click to upload"}
                      </p>
                      <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>Multiple PDFs supported</p>
                    </div>
                    <input ref={bulkFileInputRef} type="file" accept=".pdf" multiple className="hidden"
                      onChange={(e) => Array.from(e.target.files ?? []).forEach(handleBulkFileUpload)} />
                    {bulkResumes.length > 0 && (
                      <ul className="mt-3 space-y-1">
                        {bulkResumes.map((r) => (
                          <li key={r.name} className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs" style={{ background: "var(--success-bg)" }}>
                            <FileText size={12} style={{ color: "var(--success-text)" }} />
                            <span className="flex-1 truncate font-medium" style={{ color: "var(--success-text)" }}>{r.name}</span>
                            <button onClick={() => setBulkResumes((p) => p.filter((x) => x.name !== r.name))} className="p-0.5 rounded hover:opacity-70">
                              <X size={12} style={{ color: "var(--success-text)" }} />
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>

                {/* Filters */}
                <div className="rounded-[var(--radius)] border overflow-hidden" style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}>
                  <div className="px-5 py-3 border-b" style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}>
                    <h2 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>Job Filters</h2>
                  </div>
                  <div className="p-5 space-y-3">
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>Keyword</label>
                      <input value={bulkKeyword} onChange={(e) => setBulkKeyword(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                        placeholder="nurse, engineer…" />
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>Province</label>
                      <input value={bulkProvince} onChange={(e) => setBulkProvince(e.target.value.toUpperCase())}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                        placeholder="ON" />
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>City</label>
                      <input value={bulkCity} onChange={(e) => setBulkCity(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                        placeholder="Toronto" />
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>Category</label>
                      <input value={bulkCategory} onChange={(e) => setBulkCategory(e.target.value)}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                        placeholder="Engineering" />
                    </div>
                    <div>
                      <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>Max Jobs (1–50)</label>
                      <input value={bulkMaxJobs} onChange={(e) => setBulkMaxJobs(e.target.value)} type="number" min={1} max={50}
                        className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                        style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }} />
                    </div>
                    <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                      Or <button onClick={() => router.push("/jobs")} className="underline" style={{ color: "var(--accent)" }}>pick specific jobs</button> from the Job Search page.
                    </p>
                  </div>
                </div>
              </div>

              {/* Error */}
              {error && <div className="p-3 rounded-lg text-sm" style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}>{error}</div>}

              {/* Bulk submit */}
              <div className="flex justify-center">
                <button onClick={handleBulkSubmit} disabled={!canBulkSubmit}
                  className="flex items-center gap-2 px-8 py-3 rounded-lg text-sm font-semibold transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
                >
                  {loading ? <><Loader2 size={16} className="animate-spin" />Running NLP pipeline…</> : <>Run Bulk Match<ArrowRight size={16} /></>}
                </button>
              </div>
            </div>
          )}

          {/* ── SINGLE / RANK MODES ── */}
          {mode !== "bulk" && (
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
          )}

          {/* Error message (single/rank) */}
          {mode !== "bulk" && error && (
            <div
              className="mt-4 p-3 rounded-lg text-sm"
              style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}
            >
              {error}
            </div>
          )}

          {/* Analyze button (single/rank) */}
          {mode !== "bulk" && (
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
          )}
        </div>
      </main>
    </div>
  );
}
