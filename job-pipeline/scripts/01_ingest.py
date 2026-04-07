"""Download raw Canadian job and occupation datasets."""
from __future__ import annotations

from pathlib import Path

from _common import (
    ALBERTA_DATASET_ID,
    ALBERTA_PACKAGE_URL,
    JOB_BANK_DATASET_ID,
    CKAN_PACKAGE_URL,
    NOC_STRUCTURE_URL,
    OASIS_2022_DATASET_ID,
    OASIS_2025_DATASET_ID,
    RAW_DIR,
    choose_alberta_resources,
    choose_job_bank_resources,
    choose_oasis_resource,
    download_file,
    ensure_directories,
    fetch_ckan_resources,
    filename_from_resource,
    log,
)


def download_resources(resources: list[dict], destination_dir: Path, step: str) -> int:
    count = 0
    for resource in resources:
        url = resource.get("url")
        if not url:
            continue
        filename = filename_from_resource(resource)
        destination = destination_dir / filename
        log(step, f"Downloading {filename}")
        download_file(url, destination)
        count += 1
    return count


def main() -> None:
    ensure_directories()

    job_bank_dir = RAW_DIR / "job_bank"
    alberta_dir = RAW_DIR / "alberta"
    reference_dir = RAW_DIR / "reference"

    log("ingest", "Fetching Job Bank CKAN metadata")
    job_bank_resources = choose_job_bank_resources(fetch_ckan_resources(CKAN_PACKAGE_URL, JOB_BANK_DATASET_ID))
    job_bank_count = download_resources(job_bank_resources, job_bank_dir, "ingest")

    log("ingest", "Fetching Alberta CKAN metadata")
    alberta_resources = choose_alberta_resources(fetch_ckan_resources(ALBERTA_PACKAGE_URL, ALBERTA_DATASET_ID))
    alberta_count = download_resources(alberta_resources, alberta_dir, "ingest")

    log("ingest", "Downloading OaSIS reference resources")
    oasis_2022_resources = fetch_ckan_resources(CKAN_PACKAGE_URL, OASIS_2022_DATASET_ID)
    oasis_2025_resources = fetch_ckan_resources(CKAN_PACKAGE_URL, OASIS_2025_DATASET_ID)
    oasis_downloads = {
        "oasis_lead_statements.csv": choose_oasis_resource(oasis_2022_resources, ["lead statement"]),
        "oasis_main_duties.csv": choose_oasis_resource(oasis_2022_resources, ["main duties"]),
        "oasis_workplaces.csv": choose_oasis_resource(oasis_2022_resources, ["workplaces"]),
        "oasis_skills.csv": choose_oasis_resource(oasis_2025_resources, ["skills", "2025"]),
    }

    for filename, resource in oasis_downloads.items():
        log("ingest", f"Downloading {filename}")
        download_file(resource["url"], reference_dir / filename)

    log("ingest", "Attempting NOC structure download")
    try:
        download_file(NOC_STRUCTURE_URL, reference_dir / "noc_structure.csv")
    except Exception as exc:
        log("ingest", f"Warning: NOC structure download failed and will be skipped for now: {exc}")

    log("ingest", f"Completed downloads: {job_bank_count} Job Bank files, {alberta_count} Alberta files, {len(oasis_downloads)} OaSIS reference files")


if __name__ == "__main__":
    main()