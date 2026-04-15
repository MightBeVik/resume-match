"use client";

import { useRouter } from "next/navigation";
import { ArrowLeft, FileSearch, Brain, Database, Users } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";

export default function AboutPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-page)" }}>
      {/* Header */}
      <header
        className="sticky top-0 flex items-center justify-between px-8 py-4 border-b bg-[var(--bg-card)] z-10"
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
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold transition-colors"
            style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
          >
            <ArrowLeft size={14} />
            Back
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-8 py-12 space-y-12">
        {/* Hero Section */}
        <section className="space-y-4">
          <h1 className="text-4xl font-extrabold tracking-tight" style={{ color: "var(--text-primary)" }}>
            About ResumeMatch
          </h1>
          <p className="text-lg leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            ResumeMatch is an NLP-powered tool that analyzes how well your resume matches a job description. Using advanced natural language processing techniques, it provides detailed insights into your qualifications across multiple dimensions.
          </p>
        </section>

        {/* What It Is */}
        <section
          className="rounded-[var(--radius)] border p-8 space-y-4"
          style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
        >
          <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
            What It Is
          </h2>
          <p style={{ color: "var(--text-secondary)" }}>
            ResumeMatch is an academic project built for an NLP course that demonstrates practical applications of natural language processing in real-world scenarios. It's a two-part matching engine:
          </p>
          <ul className="space-y-3 ml-6" style={{ color: "var(--text-secondary)" }}>
            <li className="flex gap-3">
              <span className="font-bold" style={{ color: "var(--accent)" }}>
                Single JD Analysis:
              </span>
              <span>Compare your resume against a specific job description to identify strengths and gaps.</span>
            </li>
            <li className="flex gap-3">
              <span className="font-bold" style={{ color: "var(--accent)" }}>
                Batch Ranking:
              </span>
              <span>Rank your resume against hundreds of real job postings from the Canadian Job Bank database.</span>
            </li>
          </ul>
        </section>

        {/* How It Works */}
        <section
          className="rounded-[var(--radius)] border p-8 space-y-6"
          style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
        >
          <div className="flex items-center gap-3">
            <Brain size={32} style={{ color: "var(--accent)" }} />
            <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
              How It Works
            </h2>
          </div>

          <p style={{ color: "var(--text-secondary)" }}>
            The matching engine runs a multi-stage NLP pipeline that combines statistical and semantic techniques:
          </p>

          <div className="space-y-4">
            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                1. Preprocessing
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Tokenizes text, removes stopwords, and lemmatizes words to their base forms using NLTK.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                2. Section Parsing
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Extracts structured sections from both resume and job description (requirements, responsibilities, etc.).
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                3. Entity Extraction
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Uses spaCy for Named Entity Recognition (NER) to identify skills, organizations, degrees, and education.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                4. Keyword Extraction
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Applies TF-IDF vectorization to identify the most important terms in each document.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                5. Similarity Scoring
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Computes both TF-IDF cosine similarity and semantic similarity using sentence-transformers embeddings.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                6. Weighted Scoring
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Combines section-level scores (skills 40%, experience 25%, education 15%, preferred 10%, semantic 10%) for a final match score.
              </p>
            </div>
          </div>

          <div
            className="rounded-lg border p-4 mt-4"
            style={{ borderColor: "var(--warning)", background: "var(--warning-bg)" }}
          >
            <p className="text-sm font-semibold" style={{ color: "var(--warning-text)" }}>
              10+ NLP Techniques Demonstrated: Tokenization, stopword removal, lemmatization, sentence segmentation, NER, POS tagging, noun phrase chunking, TF-IDF vectorization, cosine similarity, and semantic embeddings.
            </p>
          </div>
        </section>

        {/* The Dataset */}
        <section
          className="rounded-[var(--radius)] border p-8 space-y-6"
          style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
        >
          <div className="flex items-center gap-3">
            <Database size={32} style={{ color: "var(--accent)" }} />
            <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
              The Dataset
            </h2>
          </div>

          <p style={{ color: "var(--text-secondary)" }}>
            The ranked job matching feature uses real job posting data from the Canadian Job Bank, enriched with occupational metadata.
          </p>

          <div className="space-y-4">
            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                Primary Source: Job Bank
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                The Government of Canada's official job posting database, covering English-language postings across all provinces and sectors.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                Occupational Classification: NOC
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Joined with National Occupational Classification (NOC) metadata to add standardized job titles, codes, and TEER levels (skill complexity tiers).
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                Competency Framework: OaSIS
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Enhanced with OaSIS (Occupational Analysis and Skills) competency data to standardize skill and requirement descriptions.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                Data Processing Pipeline
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Raw job postings are ingested, normalized, joined with classification data, and synthesized into structured job descriptions stored in SQLite for efficient ranking.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold mb-2" style={{ color: "var(--text-primary)" }}>
                Geographic Scope
              </h3>
              <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                Jobs are filterable by Canadian province and city. Filter by location to find opportunities near you.
              </p>
            </div>
          </div>
        </section>

        {/* The Team */}
        <section
          className="rounded-[var(--radius)] border p-8 space-y-6"
          style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
        >
          <div className="flex items-center gap-3">
            <Users size={32} style={{ color: "var(--accent)" }} />
            <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
              The Team
            </h2>
          </div>

          <p style={{ color: "var(--text-secondary)" }}>
            ResumeMatch is an NLP course project built by a team passionate about demonstrating practical applications of language models and machine learning in career matching.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold" style={{ color: "var(--text-primary)" }}>
                NLP & Backend Engineering
              </h3>
              <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
                FastAPI server, NLP pipeline, entity extraction, semantic similarity, and ranking engine.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold" style={{ color: "var(--text-primary)" }}>
                Frontend & UX
              </h3>
              <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
                React/Next.js interface, score visualization, and results presentation.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold" style={{ color: "var(--text-primary)" }}>
                Data Pipeline
              </h3>
              <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
                Job data ingestion, normalization, enrichment, and SQLite database creation.
              </p>
            </div>

            <div
              className="rounded-lg border p-4"
              style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
            >
              <h3 className="font-bold" style={{ color: "var(--text-primary)" }}>
                Quality Assurance
              </h3>
              <p className="text-sm mt-2" style={{ color: "var(--text-secondary)" }}>
                Unit and integration tests, end-to-end validation, and performance optimization.
              </p>
            </div>
          </div>

          <div
            className="rounded-lg border p-4 mt-6"
            style={{ borderColor: "var(--border)", background: "var(--bg-page)" }}
          >
            <p className="text-sm font-semibold mb-2" style={{ color: "var(--text-primary)" }}>
              Academic Context
            </p>
            <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
              Built as a capstone project for "Introduction to Natural Language Processing." The goal is to demonstrate understanding of core NLP techniques in a complete, deployable application.
            </p>
          </div>
        </section>

        {/* Tech Stack */}
        <section
          className="rounded-[var(--radius)] border p-8 space-y-4"
          style={{ borderColor: "var(--border)", background: "var(--bg-card)" }}
        >
          <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
            Tech Stack
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-bold mb-3" style={{ color: "var(--text-primary)" }}>
                Backend
              </h3>
              <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
                <li>• <strong>FastAPI</strong> - REST API framework</li>
                <li>• <strong>spaCy</strong> - NLP and named entity recognition</li>
                <li>• <strong>NLTK</strong> - Tokenization and lemmatization</li>
                <li>• <strong>scikit-learn</strong> - TF-IDF vectorization and metrics</li>
                <li>• <strong>sentence-transformers</strong> - Semantic embeddings</li>
                <li>• <strong>pdfplumber</strong> - PDF text extraction</li>
              </ul>
            </div>

            <div>
              <h3 className="font-bold mb-3" style={{ color: "var(--text-primary)" }}>
                Frontend
              </h3>
              <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
                <li>• <strong>Next.js 16</strong> - React framework</li>
                <li>• <strong>React 19</strong> - UI library</li>
                <li>• <strong>TypeScript</strong> - Type safety</li>
                <li>• <strong>Tailwind CSS 4</strong> - Styling</li>
                <li>• <strong>Lucide React</strong> - Icons</li>
              </ul>
            </div>

            <div className="md:col-span-2">
              <h3 className="font-bold mb-3" style={{ color: "var(--text-primary)" }}>
                Data & Infrastructure
              </h3>
              <ul className="space-y-1 text-sm" style={{ color: "var(--text-secondary)" }}>
                <li>• <strong>SQLite</strong> - Job database</li>
                <li>• <strong>Python</strong> - Data pipeline and NLP processing</li>
              </ul>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="text-center py-8 space-y-4">
          <h2 className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
            Ready to match?
          </h2>
          <button
            onClick={() => router.push("/")}
            className="inline-block px-6 py-3 rounded-lg font-semibold transition-colors"
            style={{ background: "var(--text-primary)", color: "var(--text-on-primary)" }}
          >
            Start Analyzing
          </button>
        </section>
      </main>
    </div>
  );
}
