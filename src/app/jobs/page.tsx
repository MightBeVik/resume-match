"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  Search, MapPin, X, ChevronLeft, ChevronRight,
  FileSearch, Briefcase, CheckSquare, Square, ArrowRight, Loader2,
} from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const PROVINCES = ["", "AB", "BC", "MB", "NB", "NL", "NS", "NT", "NU", "ON", "PE", "QC", "SK", "YT"];

interface JobCard {
  id: number;
  job_title: string | null;
  employer_name: string | null;
  city: string | null;
  province: string | null;
  salary: string | null;
  broad_category: string | null;
  date_posted: string | null;
  jd_preview: string | null;
}

interface SearchResponse {
  total: number;
  page: number;
  page_size: number;
  results: JobCard[];
}

export default function JobsPage() {
  const router = useRouter();

  // Filter state
  const [keyword, setKeyword] = useState("");
  const [province, setProvince] = useState("");
  const [city, setCity] = useState("");
  const [category, setCategory] = useState("");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  // Results state
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Selection state — persists across pages via card cache
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const cardCache = useRef<Map<number, JobCard>>(new Map());

  const doSearch = useCallback(async (p: number, kw: string, prov: string, ct: string, cat: string) => {
    setLoading(true);
    setError(null);
    try {
      const body = {
        keyword: kw || null,
        province: prov || null,
        city: ct || null,
        broad_category: cat || null,
        page: p,
        page_size: PAGE_SIZE,
      };
      const res = await fetch(`${API_BASE_URL}/api/jobs/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error("Search failed");
      const data: SearchResponse = await res.json();
      // cache cards
      data.results.forEach((j) => cardCache.current.set(j.id, j));
      setResults(data);
      setPage(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    doSearch(1, "", "", "", "");
  }, [doSearch]);

  const handleSearch = (e?: React.FormEvent) => {
    e?.preventDefault();
    doSearch(1, keyword, province, city, category);
  };

  const toggleId = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else if (next.size < 50) {
        next.add(id);
      }
      return next;
    });
  };

  const clearSelection = () => setSelectedIds(new Set());

  const goToLeaderboard = () => {
    const ids = [...selectedIds];
    const cards = ids.map((id) => cardCache.current.get(id)).filter(Boolean) as JobCard[];
    sessionStorage.setItem("selectedJobIds", JSON.stringify(ids));
    sessionStorage.setItem("selectedJobCards", JSON.stringify(cards));
    // Clear any previous bulk match result
    sessionStorage.removeItem("bulkMatchResult");
    router.push("/leaderboard");
  };

  const totalPages = results ? Math.ceil(results.total / PAGE_SIZE) : 0;
  const selCount = selectedIds.size;

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header
        className="flex items-center justify-between px-8 py-4 border-b bg-[var(--bg-card)] flex-shrink-0"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "var(--text-primary)" }}
          >
            <FileSearch size={16} style={{ color: "var(--text-on-primary)" }} />
          </div>
          <button
            onClick={() => router.push("/")}
            className="text-lg font-bold tracking-tight hover:opacity-80 transition-opacity"
            style={{ color: "var(--text-primary)" }}
          >
            ResumeMatch
          </button>
          <span className="text-xs ml-1" style={{ color: "var(--text-muted)" }}>/ Job Search</span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => router.push("/")}
            className="text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors"
            style={{ color: "var(--text-secondary)", background: "var(--bg-hover)" }}
          >
            ← Home
          </button>
          <ThemeToggle />
        </div>
      </header>

      {/* Body */}
      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <aside
          className="w-64 flex-shrink-0 border-r flex flex-col overflow-y-auto"
          style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
        >
          <form onSubmit={handleSearch} className="p-4 space-y-4 flex-1">
            <h2 className="text-sm font-bold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
              Filters
            </h2>

            {/* Keyword */}
            <div>
              <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                Keyword
              </label>
              <div className="relative">
                <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: "var(--text-muted)" }} />
                <input
                  value={keyword}
                  onChange={(e) => setKeyword(e.target.value)}
                  className="w-full rounded-lg border py-2 pl-8 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                  style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                  placeholder="nurse, developer…"
                />
              </div>
            </div>

            {/* Province */}
            <div>
              <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                Province
              </label>
              <select
                value={province}
                onChange={(e) => setProvince(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
              >
                {PROVINCES.map((p) => (
                  <option key={p} value={p}>{p || "All provinces"}</option>
                ))}
              </select>
            </div>

            {/* City */}
            <div>
              <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                City
              </label>
              <div className="relative">
                <MapPin size={13} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: "var(--text-muted)" }} />
                <input
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  className="w-full rounded-lg border py-2 pl-8 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                  style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                  placeholder="Toronto"
                />
              </div>
            </div>

            {/* Category */}
            <div>
              <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                Category
              </label>
              <input
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
                style={{ borderColor: "var(--border)", color: "var(--text-primary)", background: "var(--bg-page)" }}
                placeholder="Engineering"
              />
            </div>

            <button
              type="submit"
              className="w-full flex items-center justify-center gap-2 rounded-lg py-2 text-sm font-semibold transition-opacity"
              style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
            >
              <Search size={14} />
              Search
            </button>
          </form>

          {/* Selection panel */}
          <div className="border-t p-4" style={{ borderColor: "var(--border)" }}>
            {selCount === 0 ? (
              <p className="text-xs text-center" style={{ color: "var(--text-muted)" }}>
                Check job cards to select them for bulk matching
              </p>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    {selCount} / 50 selected
                  </span>
                  <button
                    onClick={clearSelection}
                    className="text-xs px-2 py-0.5 rounded hover:opacity-70"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Clear
                  </button>
                </div>
                <button
                  onClick={goToLeaderboard}
                  className="w-full flex items-center justify-center gap-2 rounded-lg py-2 text-sm font-semibold transition-opacity"
                  style={{ background: "var(--accent)", color: "#fff" }}
                >
                  Bulk Match
                  <ArrowRight size={14} />
                </button>
              </div>
            )}
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 min-h-0 overflow-y-auto p-6">
          {/* Results header */}
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-lg font-bold" style={{ color: "var(--text-primary)" }}>
                {loading ? "Searching…" : results ? `${results.total.toLocaleString()} jobs` : ""}
              </h1>
              {results && (
                <p className="text-xs" style={{ color: "var(--text-muted)" }}>
                  Page {page} of {totalPages}
                </p>
              )}
            </div>
            {loading && (
              <Loader2 size={18} className="animate-spin" style={{ color: "var(--accent)" }} />
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="mb-4 rounded-lg p-3 text-sm" style={{ background: "var(--danger-bg)", color: "var(--danger-text)" }}>
              {error}
            </div>
          )}

          {/* Job cards grid */}
          {results && results.results.length > 0 ? (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              {results.results.map((job) => {
                const sel = selectedIds.has(job.id);
                return (
                  <div
                    key={job.id}
                    onClick={() => toggleId(job.id)}
                    className="rounded-[var(--radius)] border cursor-pointer transition-all"
                    style={{
                      borderColor: sel ? "var(--accent)" : "var(--border)",
                      background: sel ? "var(--accent-light)" : "var(--bg-card)",
                      boxShadow: sel ? "0 0 0 1px var(--accent)" : undefined,
                    }}
                  >
                    <div className="p-4">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <h3
                            className="text-sm font-semibold leading-snug truncate"
                            style={{ color: "var(--text-primary)" }}
                          >
                            {job.job_title ?? "Untitled"}
                          </h3>
                          <p className="text-xs mt-0.5 truncate" style={{ color: "var(--text-secondary)" }}>
                            {job.employer_name ?? ""}
                          </p>
                        </div>
                        <div className="flex-shrink-0 mt-0.5">
                          {sel ? (
                            <CheckSquare size={18} style={{ color: "var(--accent)" }} />
                          ) : (
                            <Square size={18} style={{ color: "var(--border-strong)" }} />
                          )}
                        </div>
                      </div>

                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {job.city && (
                          <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full" style={{ background: "var(--bg-hover)", color: "var(--text-secondary)" }}>
                            <MapPin size={10} />
                            {job.city}{job.province ? `, ${job.province}` : ""}
                          </span>
                        )}
                        {job.broad_category && (
                          <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full" style={{ background: "var(--bg-hover)", color: "var(--text-secondary)" }}>
                            <Briefcase size={10} />
                            {job.broad_category}
                          </span>
                        )}
                        {job.salary && (
                          <span className="text-xs px-2 py-0.5 rounded-full font-medium" style={{ background: "var(--success-bg)", color: "var(--success-text)" }}>
                            {job.salary}
                          </span>
                        )}
                      </div>

                      {job.jd_preview && (
                        <p className="mt-2 text-xs leading-relaxed line-clamp-2" style={{ color: "var(--text-muted)" }}>
                          {job.jd_preview}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : !loading ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <Briefcase size={40} style={{ color: "var(--text-muted)" }} />
              <p className="mt-3 text-sm" style={{ color: "var(--text-muted)" }}>
                No jobs found. Try adjusting your filters.
              </p>
            </div>
          ) : null}

          {/* Pagination */}
          {results && totalPages > 1 && (
            <div className="mt-8 flex items-center justify-center gap-2">
              <button
                onClick={() => doSearch(page - 1, keyword, province, city, category)}
                disabled={page <= 1 || loading}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm font-medium disabled:opacity-40 transition-opacity"
                style={{ background: "var(--bg-card)", color: "var(--text-primary)", border: "1px solid var(--border)" }}
              >
                <ChevronLeft size={14} />
                Prev
              </button>

              <span className="text-sm px-3" style={{ color: "var(--text-secondary)" }}>
                {page} / {totalPages}
              </span>

              <button
                onClick={() => doSearch(page + 1, keyword, province, city, category)}
                disabled={page >= totalPages || loading}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm font-medium disabled:opacity-40 transition-opacity"
                style={{ background: "var(--bg-card)", color: "var(--text-primary)", border: "1px solid var(--border)" }}
              >
                Next
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
