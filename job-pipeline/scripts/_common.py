"""Shared helpers for the Canadian jobs pipeline."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "jobs.db"

JOB_BANK_DATASET_ID = "ea639e28-c0fc-48bf-b5dd-b8899bd43072"
ALBERTA_DATASET_ID = "20f6ac7c-0a56-4b77-a3cc-3b9adae24425"

CKAN_PACKAGE_URL = "https://open.canada.ca/data/api/action/package_show?id={dataset_id}"
ALBERTA_PACKAGE_URL = "https://open.alberta.ca/api/action/package_show?id={dataset_id}"

NOC_STRUCTURE_URL = "https://www.statcan.gc.ca/eng/statistical-programs/document/NOC-CNP-2021-Structure-V1-eng.csv"
OASIS_LEAD_STATEMENTS_URL = "https://open.canada.ca/data/dataset/eeb3e442-9f19-4d12-8b38-c488fe4f6e5e/resource/f1b0b3c4-8982-4104-ba96-794044ad65e8/download/lead-statement_oasis_2022_v1.0.csv"
OASIS_WORKPLACES_URL = "https://open.canada.ca/data/dataset/eeb3e442-9f19-4d12-8b38-c488fe4f6e5e/resource/c3e16f83-5d09-43be-b1b4-49526cc25ed8/download/workplaces-employers_oasis_2022_v1.0.csv"
OASIS_TAXONOMY_URL = "https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/2a7a17bf-b67c-4fc2-b636-959c7f1d7ae4/download/guide_oasis_2025_v4.0.csv"
OASIS_2022_DATASET_ID = "eeb3e442-9f19-4d12-8b38-c488fe4f6e5e"
OASIS_2025_DATASET_ID = "10ce43bd-fb58-4969-806b-4bffebc87bec"


def log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", True):
        raise RuntimeError(f"API request failed for {url}: {json.dumps(payload)[:500]}")
    return payload


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "download"


def filename_from_resource(resource: dict[str, Any]) -> str:
        url = resource.get("url") or ""
        parsed_path = Path(urlparse(url).path)
        extension = parsed_path.suffix.lower()
        filename = sanitize_filename(resource.get("name") or parsed_path.stem or "download")
        if extension and not filename.lower().endswith(extension):
            filename += extension
        return filename


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=300, stream=True)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return destination


def fetch_ckan_resources(api_url: str, dataset_id: str) -> list[dict[str, Any]]:
    payload = fetch_json(api_url.format(dataset_id=dataset_id))
    return payload["result"].get("resources", [])


def choose_job_bank_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for resource in resources:
        url = (resource.get("url") or "").lower()
        language = resource.get("language") or []
        normalized_languages = {str(item).lower() for item in language}
        if url.endswith(".csv") and ("en" in normalized_languages or "english" in normalized_languages or "-en-" in url):
            selected.append(resource)
    if not selected:
        raise RuntimeError("No English Job Bank CSV resources were found in the CKAN dataset response.")

    resource_limit = int(os.getenv("JOB_BANK_RESOURCE_LIMIT", "2"))
    resource_limit = max(1, min(resource_limit, len(selected)))
    return selected[:resource_limit]


def choose_alberta_resources(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for resource in resources:
        url = (resource.get("url") or "").lower()
        if url.endswith(".csv") or url.endswith(".xlsx") or url.endswith(".xls") or url.endswith(".pdf"):
            selected.append(resource)
    if not selected:
        raise RuntimeError("No Alberta vacancy resources were found in the CKAN dataset response.")
    return selected[:2]


def choose_oasis_resource(resources: list[dict[str, Any]], required_terms: list[str]) -> dict[str, Any]:
    lowered_terms = [term.lower() for term in required_terms]
    for resource in resources:
        name = (resource.get("name") or "").lower()
        format_name = (resource.get("format") or "").lower()
        language = {str(item).lower() for item in (resource.get("language") or [])}
        if format_name != "csv":
            continue
        if language and "en" not in language and "en-ca [default]" not in language and "english" not in language:
            continue
        if all(term in name for term in lowered_terms):
            return resource
    raise RuntimeError(f"Could not find OaSIS resource matching terms: {required_terms}")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("rb") as handle:
            prefix = handle.read(4)
        if prefix.startswith(b"\xff\xfe") or prefix.startswith(b"\xfe\xff"):
            return pd.read_csv(path, encoding="utf-16", sep="\t")
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported tabular file type: {path}")


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: normalize_column_name(column) for column in frame.columns}
    return frame.rename(columns=renamed)


def detect_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        key = normalize_column_name(candidate)
        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name
    return None


def normalize_noc_code(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return None
    if len(digits) == 4:
        return digits.zfill(5)
    if len(digits) >= 5:
        return digits[:5]
    return digits.zfill(5)


def first_non_empty(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return row[column]
    return None