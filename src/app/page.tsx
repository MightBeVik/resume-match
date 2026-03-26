"use client";

import { useState } from "react";
import { Header } from "@/components/header";
import { MatcherEmptyState } from "@/components/matcher/empty-state";
import { MatcherLoadingState } from "@/components/matcher/loading-state";
import { MatcherResultsState } from "@/components/matcher/results-state";
import { RewriteInputState } from "@/components/rewrite/input-state";
import { RewriteLoadingState } from "@/components/rewrite/loading-state";
import { RewriteResultsState } from "@/components/rewrite/results-state";

type AppState = "empty" | "loading" | "results";

const mockMatchResults = {
  matchScore: 82,
  verdict: "Strong Match",
  verdictDescription: "This resume aligns well with the job requirements",
  skills: [
    { name: "React", matched: true },
    { name: "TypeScript", matched: true },
    { name: "Tailwind CSS", matched: true },
    { name: "REST APIs", matched: true },
    { name: "Git", matched: true },
    { name: "GraphQL", matched: false },
    { name: "AWS", matched: false },
    { name: "CI/CD", matched: false },
  ],
  education: [
    { name: "B.S. Computer Science", matched: true },
    { name: "4+ Years Degree", matched: true },
    { name: "M.S. Preferred", matched: false },
  ],
  experience: [
    { name: "5+ Years Frontend", matched: true },
    { name: "Team Lead", matched: true },
    { name: "Agile/Scrum", matched: true },
    { name: "10+ Years Total", matched: false },
    { name: "Enterprise SaaS", matched: false },
  ],
};

export default function Home() {
  const [activeTab, setActiveTab] = useState<"matcher" | "rewrite">("matcher");
  const [matcherState, setMatcherState] = useState<AppState>("empty");
  const [rewriteState, setRewriteState] = useState<AppState>("empty");

  const handleAnalyze = () => {
    setMatcherState("loading");
    setTimeout(() => setMatcherState("results"), 2500);
  };

  const handleRewrite = () => {
    setRewriteState("loading");
    setTimeout(() => setRewriteState("results"), 3000);
  };

  const handleStartOver = (feature: "matcher" | "rewrite") => {
    if (feature === "matcher") setMatcherState("empty");
    else setRewriteState("empty");
  };

  return (
    <div className="h-full flex flex-col">
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="flex-1 min-h-0 overflow-auto">
        {activeTab === "matcher" && (
          <>
            {matcherState === "empty" && (
              <MatcherEmptyState onAnalyze={handleAnalyze} />
            )}
            {matcherState === "loading" && <MatcherLoadingState />}
            {matcherState === "results" && (
              <MatcherResultsState
                results={mockMatchResults}
                onRewrite={() => {
                  setActiveTab("rewrite");
                  handleRewrite();
                }}
                onStartOver={() => handleStartOver("matcher")}
              />
            )}
          </>
        )}

        {activeTab === "rewrite" && (
          <>
            {rewriteState === "empty" && (
              <RewriteInputState onRewrite={handleRewrite} />
            )}
            {rewriteState === "loading" && <RewriteLoadingState />}
            {rewriteState === "results" && (
              <RewriteResultsState
                onStartOver={() => handleStartOver("rewrite")}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
