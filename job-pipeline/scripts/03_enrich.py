"""Join postings to NOC and OaSIS competencies."""
from __future__ import annotations

import pandas as pd

from _common import PROCESSED_DIR, detect_column, ensure_directories, log, normalize_noc_code


SKILL_CATEGORY_HINTS = {"skills", "abilities", "personal attributes", "work activities"}


def _read_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed input {path}. Run the earlier pipeline steps first.")
    frame = pd.read_csv(path)
    if "noc_code" in frame.columns:
        frame["noc_code"] = frame["noc_code"].map(normalize_noc_code)
    return frame


def _aggregate_profile_components(lead_frame: pd.DataFrame, duties_frame: pd.DataFrame) -> pd.DataFrame:
    lead_noc_column = detect_column(lead_frame, ["noc_code", "noc", "oasis_profile_code", "code_oasis"])
    lead_column = detect_column(lead_frame, ["lead_statement", "lead_statements"])
    if not lead_noc_column or not lead_column:
        raise RuntimeError("Could not identify NOC and lead statement columns in OaSIS lead statements data.")

    duties_noc_column = detect_column(duties_frame, ["noc_code", "noc", "oasis_profile_code", "code_oasis"])
    duties_column = detect_column(duties_frame, ["main_duties", "main_duty", "duty", "duties"])
    if not duties_noc_column or not duties_column:
        raise RuntimeError("Could not identify NOC and main duties columns in OaSIS main duties data.")

    lead_subset = pd.DataFrame(
        {
            "noc_code": lead_frame[lead_noc_column].map(normalize_noc_code),
            "lead_statement": lead_frame[lead_column],
        }
    ).dropna(subset=["noc_code"])
    lead_subset = lead_subset.groupby("noc_code", as_index=False).agg(
        {
            "lead_statement": lambda values: " ".join(dict.fromkeys(str(value).strip() for value in values if pd.notna(value) and str(value).strip())),
        }
    )

    duties_subset = pd.DataFrame(
        {
            "noc_code": duties_frame[duties_noc_column].map(normalize_noc_code),
            "main_duties": duties_frame[duties_column],
        }
    ).dropna(subset=["noc_code"])
    duties_subset = duties_subset.groupby("noc_code", as_index=False).agg(
        {
            "main_duties": lambda values: "\n".join(dict.fromkeys(str(value).strip() for value in values if pd.notna(value) and str(value).strip())),
        }
    )

    return lead_subset.merge(duties_subset, on="noc_code", how="outer")


def _aggregate_workplaces(frame: pd.DataFrame) -> pd.DataFrame:
    noc_column = detect_column(frame, ["noc_code", "noc", "oasis_profile_code", "code_oasis"])
    workplaces_column = detect_column(frame, ["workplaces", "employers", "workplace", "employer_type"])
    if not noc_column or not workplaces_column:
        raise RuntimeError("Could not identify NOC and workplaces columns in OaSIS workplaces data.")

    subset = pd.DataFrame(
        {
            "noc_code": frame[noc_column].map(normalize_noc_code),
            "workplaces": frame[workplaces_column],
        }
    ).dropna(subset=["noc_code"])

    return subset.groupby("noc_code", as_index=False).agg(
        {
            "workplaces": lambda values: " | ".join(dict.fromkeys(str(value).strip() for value in values if pd.notna(value) and str(value).strip()))
        }
    )


def _aggregate_taxonomy(frame: pd.DataFrame) -> pd.DataFrame:
    noc_column = detect_column(frame, ["noc_code", "noc", "oasis_profile_code", "code_oasis"])
    title_column = detect_column(frame, ["oasis_2021_labels", "label", "title", "occupation_title"])
    if not noc_column:
        raise RuntimeError("Could not identify the NOC/OaSIS profile code column in OaSIS skills data.")

    metric_columns = [column for column in frame.columns if column not in {noc_column, title_column}]
    if not metric_columns:
        raise RuntimeError("No skill descriptor columns were found in OaSIS skills data.")

    subset = frame.melt(
        id_vars=[column for column in [noc_column, title_column] if column],
        value_vars=metric_columns,
        var_name="competency_label",
        value_name="importance_score",
    )
    subset = subset.rename(columns={noc_column: "noc_code"})
    subset["noc_code"] = subset["noc_code"].map(normalize_noc_code)
    subset["category"] = "skills"
    subset["importance_score"] = pd.to_numeric(subset["importance_score"], errors="coerce")
    subset = subset.dropna(subset=["noc_code", "importance_score"])
    subset = subset[subset["importance_score"] > 0]
    subset["competency_label"] = subset["competency_label"].astype(str).str.replace("_", " ").str.strip()
    subset = subset[subset["competency_label"].ne("")]

    subset.to_csv(PROCESSED_DIR / "oasis_competencies_long.csv", index=False)

    ranked = subset.sort_values(["noc_code", "importance_score"], ascending=[True, False]).copy()
    ranked["category_normalized"] = ranked["category"].str.lower()
    ranked = ranked[ranked["category_normalized"].isin(SKILL_CATEGORY_HINTS) | ranked["category_normalized"].eq("skills")]

    top_skills = ranked.groupby("noc_code").head(10)
    aggregated = top_skills.groupby("noc_code", as_index=False).agg(
        {
            "competency_label": lambda values: "|".join(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
        }
    ).rename(columns={"competency_label": "top_skills"})

    return aggregated


def main() -> None:
    ensure_directories()

    raw_postings = _read_csv("raw_postings.csv")
    noc_structure = _read_csv("noc_structure.csv")
    lead_statements = _aggregate_profile_components(
        _read_csv("oasis_lead_statements.csv"),
        _read_csv("oasis_main_duties.csv"),
    )
    workplaces = _aggregate_workplaces(_read_csv("oasis_workplaces.csv"))
    top_skills = _aggregate_taxonomy(_read_csv("oasis_skills.csv"))

    oasis_competencies = top_skills.merge(workplaces, on="noc_code", how="outer")
    oasis_competencies.to_csv(PROCESSED_DIR / "oasis_competencies.csv", index=False)

    enriched = raw_postings.merge(noc_structure, on="noc_code", how="left")
    if enriched["noc_title"].notna().sum() == 0:
        log("enrich", "NOC structure reference produced zero mapped rows; continuing with Job Bank and OaSIS enrichment only")

    enriched = enriched.merge(lead_statements, on="noc_code", how="left")
    enriched = enriched.merge(oasis_competencies, on="noc_code", how="left")

    required_matches = enriched[["lead_statement", "top_skills"]].notna().any(axis=1).sum()
    if required_matches == 0:
        raise RuntimeError("JOIN from postings to OaSIS enrichment produced zero usable rows.")

    enriched.to_csv(PROCESSED_DIR / "enriched_jobs.csv", index=False)
    log("enrich", f"Wrote {len(enriched)} enriched postings")


if __name__ == "__main__":
    main()